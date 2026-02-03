/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_interleave_grad_proto.h
 * \brief
 */

#include <iostream>
#include <vector>
#include <thread>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "log/log.h"
#include "graph/graph.h"
#include "kernel_run_context_facker.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../op_host/qkv_rms_norm_rope_cache_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

int batch_size;
int seq_len;
int Nqkv;
int Nq;
int Nk;
int Nv;
int dim;
int block_num;
int block_size;

class QkvRmsNormRopeCacheTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "QkvRmsNormRopeCacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QkvRmsNormRopeCacheTiling TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream& tiling_data)
{
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

template <typename T>
static string to_string(void *buf, size_t size)
{
    std::string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_bath_quantA)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gammaQ_shape = {{dim}, {dim}};
    gert::StorageShape gammaK_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gammaQ_shape, &gammaK_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_bath_quant_AS)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gammaQ_shape = {{dim}, {dim}};
    gert::StorageShape gammaK_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};
    gert::StorageShape kOffset_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vOffset_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gammaQ_shape, &gammaK_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, &kOffset_shape, &vOffset_shape})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_bath_quantB)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 11898;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gammaQ_shape = {{dim}, {dim}};
    gert::StorageShape gammaK_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gammaQ_shape, &gammaK_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_bath_quantC)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 16;
    seq_len = 3;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 11898;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gammaQ_shape = {{dim}, {dim}};
    gert::StorageShape gammaK_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gammaQ_shape, &gammaK_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_k_quant)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gammaQ_shape = {{dim}, {dim}};
    gert::StorageShape gammaK_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 16, block_size, 16}, {block_num, Nv * dim / 16, block_size, 16}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 16, block_size, 16}, {block_num, Nv * dim / 16, block_size, 16}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gammaQ_shape, &gammaK_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, nullptr, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_v_quant)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gammaQ_shape = {{dim}, {dim}};
    gert::StorageShape gammaK_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 16, block_size, 16}, {block_num, Nk * dim / 16, block_size, 16}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 16, block_size, 16}, {block_num, Nk * dim / 16, block_size, 16}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gammaQ_shape, &gammaK_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, nullptr, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_no_quant)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gammaQ_shape = {{dim}, {dim}};
    gert::StorageShape gammaK_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 16, block_size, 16}, {block_num, Nk * dim / 16, block_size, 16}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 16, block_size, 16}, {block_num, Nv * dim / 16, block_size, 16}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 16, block_size, 16}, {block_num, Nk * dim / 16, block_size, 16}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 16, block_size, 16}, {block_num, Nv * dim / 16, block_size, 16}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gammaQ_shape, &gammaK_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_bath_quant_small)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 4;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gammaQ_shape = {{dim}, {dim}};
    gert::StorageShape gammaK_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gammaQ_shape, &gammaK_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_k_quant_small)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 4;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gammaQ_shape = {{dim}, {dim}};
    gert::StorageShape gammaK_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 16, block_size, 16}, {block_num, Nv * dim / 16, block_size, 16}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 16, block_size, 16}, {block_num, Nv * dim / 16, block_size, 16}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gammaQ_shape, &gammaK_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, nullptr, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_v_quant_small)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 4;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gammaQ_shape = {{dim}, {dim}};
    gert::StorageShape gammaK_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 16, block_size, 16}, {block_num, Nk * dim / 16, block_size, 16}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 16, block_size, 16}, {block_num, Nk * dim / 16, block_size, 16}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gammaQ_shape, &gammaK_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, nullptr, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_no_quant_small)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 4;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gammaQ_shape = {{dim}, {dim}};
    gert::StorageShape gammaK_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 16, block_size, 16}, {block_num, Nk * dim / 16, block_size, 16}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 16, block_size, 16}, {block_num, Nv * dim / 16, block_size, 16}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 16, block_size, 16}, {block_num, Nk * dim / 16, block_size, 16}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 16, block_size, 16}, {block_num, Nv * dim / 16, block_size, 16}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gammaQ_shape, &gammaK_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qkv_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {nullptr, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qkv_dimSize_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size + 2, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qkv_dtype_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qkvDim_32_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 129;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_gamma_dtype_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_gamma_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{1, dim}, {1, dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_gamma_dimOne_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{72}, {72}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_cos_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, nullptr, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_cos_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{1 * seq_len, dim}, {1 * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_cos_dtype_diff_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_sin_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, nullptr, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_sin_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{dim}, {dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_sin_dtype_diff_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_index_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, nullptr, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_index_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len + 2}, {batch_size * seq_len + 2}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_index_dtype_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qout_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, nullptr, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qout_dtype_diff_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qout_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim + 2}, {batch_size * seq_len, Nq * dim + 2}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_kcache_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, nullptr,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_kcache_dtype_diff_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_kcache_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 16, block_size, 16}, {block_num, Nk * dim / 16, block_size, 16}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_kcache_blocksize_32_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 129;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vcache_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           nullptr, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vcache_dtype_diff_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vcache_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num + 1, Nv * dim / 32, block_size, 32}, {block_num + 1, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vcache_dims_blocksize_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size + 1, 32}, {block_num, Nv * dim / 32, block_size + 1, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_kscale_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, nullptr, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_kscale_dtype_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_kscale_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{10, dim}, {10, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vscale_is_None_int8_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, nullptr, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vscale_is_notNone_fp16_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vscale_dtype_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vscale_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{dim}, {dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qkv_size_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qkv_size_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, -2};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qkv_size_dims_relation_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 2;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_headNums_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3,ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_headNums_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq + 2, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_headNums_shape_wrong1)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 3;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq + 2, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_headNums_shape_wrong2)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 3;
    Nv = 3;
    dim = 128;
    block_num = 72;
    block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gamma_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    std::map<std::string, std::string> soc_version = {{"Short_SoC_version", "Ascend910B"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::QkvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("QkvRmsNormRopeCache");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq + 2, Nk, Nv};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("QkvRmsNormRopeCache")
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gamma_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // kcache
                      .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND) // vcache
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"qkv_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}