/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <thread>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include <functional>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_chunk_gated_delta_rule.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class aclnnChunkGatedDeltaRule_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "aclnnChunkGatedDeltaRule_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "aclnnChunkGatedDeltaRule_test TearDown" << endl;
    }
};

class aclnnChunkGatedDeltaRule_test_case {
public:
    uint32_t bs = 2;
    uint32_t seqLen = 100;
    uint32_t t = bs * seqLen; 
    uint32_t nk = 4;
    uint32_t nv = 8;
    uint32_t dk = 128;
    uint32_t dv = 128;

    TensorDesc initialState;
    TensorDesc query;
    TensorDesc key;
    TensorDesc value;
    TensorDesc beta;
    TensorDesc actualSeqLengths;
    TensorDesc gOptional;
    TensorDesc out;
    TensorDesc finalState;

    void ChunkGatedDeltaRuleTestCase(int validIdx, int nullIdx)
    {
        constexpr int32_t kValueLow = 0;
        constexpr int32_t kValueHigh = 1;
        // actualSeqLengths: [seqLen, 2*seqLen, ..., bs*seqLen] 的等差序列
        vector<int32_t> actualSeqLens(bs, 0);
        for (uint32_t i = 0; i < bs; ++i) {
            actualSeqLens[i] = static_cast<int32_t>((i + 1) * seqLen);
        }

        initialState = TensorDesc({bs, nv, dv, dk}, ACL_BF16, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        query = TensorDesc({t, nk, dk}, ACL_BF16, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        key = TensorDesc({t, nk, dk}, ACL_BF16, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        value = TensorDesc({t, nv, dv}, ACL_BF16, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        beta = TensorDesc({t, nv}, ACL_BF16, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        actualSeqLengths = TensorDesc({bs}, ACL_INT32, ACL_FORMAT_ND).Value(actualSeqLens);
        gOptional = TensorDesc({t, nv}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        out = TensorDesc({t, nv, dv}, ACL_BF16, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        finalState = TensorDesc({t, nv, dv, dk}, ACL_BF16, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);

        // 非法 dtype 注入映射：index=0 表示全部 dtype 合法（序号与 validIdx 对应）
        const std::array<std::function<void()>, 10> invalidDtypeCases = {
            []() {}, // 0: 不修改（全部 dtype 合法）
            [this, kValueLow, kValueHigh]() { // 1: initialState -> ACL_INT8
                initialState = TensorDesc({bs, nv, dv, dk}, ACL_INT8, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
            },
            [this, kValueLow, kValueHigh]() { // 2: query -> ACL_INT8
                query = TensorDesc({t, nk, dk}, ACL_INT8, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
            },
            [this, kValueLow, kValueHigh]() { // 3: key -> ACL_INT8
                key = TensorDesc({t, nk, dk}, ACL_INT8, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
            },
            [this, kValueLow, kValueHigh]() { // 4: value -> ACL_INT8
                value = TensorDesc({t, nv, dv}, ACL_INT8, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
            },
            [this, kValueLow, kValueHigh]() { // 5: beta -> ACL_INT8
                beta = TensorDesc({t, nv}, ACL_INT8, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
            },
            [this, kValueLow, kValueHigh]() { // 6: actualSeqLengths -> ACL_INT8
                actualSeqLengths = TensorDesc({bs}, ACL_INT8, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
            },
            [this, kValueLow, kValueHigh]() { // 7: gOptional -> ACL_INT8
                gOptional = TensorDesc({t, nv}, ACL_INT8, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
            },
            [this, kValueLow, kValueHigh]() { // 8: out -> ACL_INT8
                out = TensorDesc({t, nv, dv}, ACL_INT8, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
            },
            [this, kValueLow, kValueHigh]() { // 9: finalState -> ACL_INT8
                finalState = TensorDesc({bs, nv, dv, dk}, ACL_INT8, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
            },
        };
        if (validIdx >= 0 && validIdx < static_cast<int>(invalidDtypeCases.size())) {
            invalidDtypeCases[validIdx]();
        }

        aclnnStatus aclRet = utTest(nullIdx);
        // 仅 gOptional 允许为空（nullIdx == 7）时仍视为合法
        aclnnStatus expected =
            ((nullIdx == 0 || nullIdx == 7) && validIdx == 0) ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID;
        EXPECT_EQ(aclRet, expected);
    }

    aclnnStatus utTest(int nullIdx)
    {
        uint64_t workspace_size = 0;
        constexpr float ScaleValue = 1.0f;
        // 通过 nullIdx 表驱动置空不同的输入/输出位置（序号与 nullIdx 对应）
        const std::array<std::function<aclnnStatus()>, 11> nullCases = {
            [this, &workspace_size]() {
                // 0: 全部非空（基准正常 case）
                auto ut = OP_API_UT(aclnnChunkGatedDeltaRule,
                                    INPUT(query, key, value, beta, initialState, actualSeqLengths, gOptional, ScaleValue),
                                    OUTPUT(out, finalState));
                return ut.TestGetWorkspaceSize(&workspace_size);
            },
            [this, &workspace_size]() {
                // 1: query 置空
                auto ut = OP_API_UT(aclnnChunkGatedDeltaRule,
                                    INPUT(nullptr, key, value, beta, initialState, actualSeqLengths, gOptional, ScaleValue),
                                    OUTPUT(out, finalState));
                return ut.TestGetWorkspaceSize(&workspace_size);
            },
            [this, &workspace_size]() {
                // 2: key 置空
                auto ut = OP_API_UT(aclnnChunkGatedDeltaRule,
                                    INPUT(query, nullptr, value, beta, initialState, actualSeqLengths, gOptional, ScaleValue),
                                    OUTPUT(out, finalState));
                return ut.TestGetWorkspaceSize(&workspace_size);
            },
            [this, &workspace_size]() {
                // 3: value 置空
                auto ut = OP_API_UT(aclnnChunkGatedDeltaRule,
                                    INPUT(query, key, nullptr, beta, initialState, actualSeqLengths, gOptional, ScaleValue),
                                    OUTPUT(out, finalState));
                return ut.TestGetWorkspaceSize(&workspace_size);
            },
            [this, &workspace_size]() {
                // 4: beta 置空
                auto ut = OP_API_UT(aclnnChunkGatedDeltaRule,
                                    INPUT(query, key, value, nullptr, initialState, actualSeqLengths, gOptional, ScaleValue),
                                    OUTPUT(out, finalState));
                return ut.TestGetWorkspaceSize(&workspace_size);
            },
            [this, &workspace_size]() {
                // 5: initialState 置空
                auto ut = OP_API_UT(aclnnChunkGatedDeltaRule,
                                    INPUT(query, key, value, beta, nullptr, actualSeqLengths, gOptional, ScaleValue),
                                    OUTPUT(out, finalState));
                return ut.TestGetWorkspaceSize(&workspace_size);
            },
            [this, &workspace_size]() {
                // 6: actualSeqLengths 置空
                auto ut = OP_API_UT(aclnnChunkGatedDeltaRule,
                                    INPUT(query, key, value, beta, initialState, nullptr, gOptional, ScaleValue),
                                    OUTPUT(out, finalState));
                return ut.TestGetWorkspaceSize(&workspace_size);
            },
            [this, &workspace_size]() {
                // 7: gOptional 置空（唯一允许为空的可选输入）
                auto ut = OP_API_UT(aclnnChunkGatedDeltaRule,
                                    INPUT(query, key, value, beta, initialState, actualSeqLengths, nullptr, ScaleValue),
                                    OUTPUT(out, finalState));
                return ut.TestGetWorkspaceSize(&workspace_size);
            },
            [this, &workspace_size]() {
                // 8: out 置空
                auto ut = OP_API_UT(aclnnChunkGatedDeltaRule,
                                    INPUT(query, key, value, beta, initialState, actualSeqLengths, gOptional, ScaleValue),
                                    OUTPUT(nullptr, finalState));
                return ut.TestGetWorkspaceSize(&workspace_size);
            },
            [this, &workspace_size]() {
                // 9: finalState 置空
                auto ut = OP_API_UT(aclnnChunkGatedDeltaRule,
                                    INPUT(query, key, value, beta, initialState, actualSeqLengths, gOptional, ScaleValue),
                                    OUTPUT(out, nullptr));
                return ut.TestGetWorkspaceSize(&workspace_size);
            },
            [this, &workspace_size]() {
                // 10: out 与 finalState 同时置空
                auto ut = OP_API_UT(aclnnChunkGatedDeltaRule,
                                    INPUT(query, key, value, beta, initialState, actualSeqLengths, gOptional, ScaleValue),
                                    OUTPUT(nullptr, nullptr));
                return ut.TestGetWorkspaceSize(&workspace_size);
            },
        };
        const size_t idx =
            (nullIdx >= 0 && nullIdx < static_cast<int>(nullCases.size())) ? static_cast<size_t>(nullIdx)
                                                                          : (nullCases.size() - 1);
        return nullCases[idx]();
    }
};

aclnnChunkGatedDeltaRule_test_case test;
// case 0-9: validIdx=0~9, nullIdx=0（覆盖各输入/输出 dtype 非法）
// case 10-19: validIdx=0，nullIdx=1~10（覆盖各输入/输出为 nullptr）
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case0)
{
    test.ChunkGatedDeltaRuleTestCase(0, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case1)
{
    test.ChunkGatedDeltaRuleTestCase(1, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case2)
{
    test.ChunkGatedDeltaRuleTestCase(2, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case3)
{
    test.ChunkGatedDeltaRuleTestCase(3, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case4)
{
    test.ChunkGatedDeltaRuleTestCase(4, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case5)
{
    test.ChunkGatedDeltaRuleTestCase(5, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case6)
{
    test.ChunkGatedDeltaRuleTestCase(6, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case7)
{
    test.ChunkGatedDeltaRuleTestCase(7, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case8)
{
    test.ChunkGatedDeltaRuleTestCase(8, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case9)
{
    test.ChunkGatedDeltaRuleTestCase(9, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case10)
{
    test.ChunkGatedDeltaRuleTestCase(0, 1);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case11)
{
    test.ChunkGatedDeltaRuleTestCase(0, 2);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case12)
{
    test.ChunkGatedDeltaRuleTestCase(0, 3);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case13)
{
    test.ChunkGatedDeltaRuleTestCase(0, 4);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case14)
{
    test.ChunkGatedDeltaRuleTestCase(0, 5);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case15)
{
    test.ChunkGatedDeltaRuleTestCase(0, 6);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case16)
{
    test.ChunkGatedDeltaRuleTestCase(0, 7);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case17)
{
    test.ChunkGatedDeltaRuleTestCase(0, 8);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case18)
{
    test.ChunkGatedDeltaRuleTestCase(0, 9);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case19)
{
    test.ChunkGatedDeltaRuleTestCase(0, 10);
}
