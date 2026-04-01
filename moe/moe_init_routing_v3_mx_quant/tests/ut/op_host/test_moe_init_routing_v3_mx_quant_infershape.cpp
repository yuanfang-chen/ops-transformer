/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "register/op_impl_registry.h"
#include "tests/utils/op_host_test_util.h"

class MoeInitRoutingV3MxQuantInfershape : public testing::Test {};

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_scalar_bf16) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_1d_bf16) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_2d_bf16) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{128, 256}, {128, 256}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{128, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_3d_bf16) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{4, 3, 4}, {4, 3, 4}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_4d_bf16) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_single_element_bf16) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_aligned_boundary_bf16) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{256, 256}, {256, 256}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{256, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_unaligned_bf16) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{33, 17}, {33, 17}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{33, 17}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_scalar_fp8_e4m3fn) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_1d_fp8_e4m3fn) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_2d_fp8_e4m3fn) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{128, 256}, {128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{128, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_3d_fp8_e4m3fn) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_4d_fp8_e4m3fn) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_single_element_fp8_e4m3fn) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_aligned_boundary_fp8_e4m3fn) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{256, 256}, {256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{256, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_unaligned_fp8_e4m3fn) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{33, 17}, {33, 17}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{33, 17}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_scalar_fp8_e5m2) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_1d_fp8_e5m2) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_2d_fp8_e5m2) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{128, 256}, {128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{128, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_3d_fp8_e5m2) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_4d_fp8_e5m2) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_single_element_fp8_e5m2) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_aligned_boundary_fp8_e5m2) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{256, 256}, {256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{256, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_unaligned_fp8_e5m2) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{33, 17}, {33, 17}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{33, 17}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_scalar_fp8_e8m0) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_1d_fp8_e8m0) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_2d_fp8_e8m0) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{128, 256}, {128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{128, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_3d_fp8_e8m0) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_4d_fp8_e8m0) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_single_element_fp8_e8m0) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_aligned_boundary_fp8_e8m0) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{256, 256}, {256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{256, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_unaligned_fp8_e8m0) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{33, 17}, {33, 17}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{33, 17}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_scalar_int32) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_1d_int32) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_2d_int32) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{128, 256}, {128, 256}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{128, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_3d_int32) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{4, 3, 4}, {4, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_4d_int32) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_single_element_int32) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_aligned_boundary_int32) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{256, 256}, {256, 256}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{256, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_unaligned_int32) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{33, 17}, {33, 17}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{33, 17}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3MxQuantInfershape, infershape_unsupported_int8) {
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3MxQuant",
        {{{32, 64}, {32, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{}}, ge::DT_INT8, ge::FORMAT_ND}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
