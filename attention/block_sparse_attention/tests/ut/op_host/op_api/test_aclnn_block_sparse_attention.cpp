/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_block_sparse_attention.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_block_sparse_attention_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_block_sparse_attention_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_block_sparse_attention_test TearDown" << endl;
    }
};

TEST_F(l2_block_sparse_attention_test, Ascend910B2_block_sparse_attention_tnd_fp16)
{
    size_t batch = 2;
    size_t numHeads = 32;
    size_t kvHeads = 8;
    size_t embeddingSize = 128;
    size_t qSeqlen = 1024;
    size_t kvSeqlen = 1024;
    size_t blockSize = 128;
    
    auto query = TensorDesc({qSeqlen * batch, numHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kvSeqlen * batch, kvHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kvSeqlen * batch, kvHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMaskOptional = TensorDesc({batch, numHeads, qSeqlen / blockSize, kvSeqlen / blockSize}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto attenMaskOptional = nullptr;
    auto blockShapeOptional = IntArrayDesc({blockSize, blockSize});
    auto actualSeqLengthsOptional = IntArrayDesc({static_cast<int64_t>(qSeqlen), static_cast<int64_t>(qSeqlen)});
    auto actualSeqLengthsKvOptional = IntArrayDesc({static_cast<int64_t>(kvSeqlen), static_cast<int64_t>(kvSeqlen)});
    auto blockTableOptional = nullptr;
    
    char qInputLayout[] = "TND";
    char kvInputLayout[] = "TND";
    int64_t numKeyValueHeads = kvHeads;
    int64_t maskType = 0;
    double scaleValue = 1.0 / std::sqrt(embeddingSize);
    int64_t innerPrecise = 1;
    int64_t blockSizeAttr = blockSize;
    int64_t preTokens = 65536;
    int64_t nextTokens = 65536;
    int64_t softmaxLseFlag = 0;
    
    auto attentionOut = TensorDesc({qSeqlen * batch, numHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto softmaxLseOptional = TensorDesc({qSeqlen * batch, numHeads, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnBlockSparseAttentionGetWorkspaceSize,
                        INPUT(query, key, value, blockSparseMaskOptional, attenMaskOptional, blockShapeOptional,
                              actualSeqLengthsOptional, actualSeqLengthsKvOptional, blockTableOptional,
                              qInputLayout, kvInputLayout, numKeyValueHeads, maskType, scaleValue,
                              innerPrecise, blockSizeAttr, preTokens, nextTokens, softmaxLseFlag),
                        OUTPUT(attentionOut, softmaxLseOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_block_sparse_attention_test, Ascend910B2_block_sparse_attention_bnsd_fp16)
{
    size_t batch = 2;
    size_t numHeads = 32;
    size_t kvHeads = 8;
    size_t embeddingSize = 128;
    size_t qSeqlen = 1024;
    size_t kvSeqlen = 1024;
    size_t blockSize = 128;
    
    auto query = TensorDesc({batch, numHeads, qSeqlen, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, kvHeads, kvSeqlen, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, kvHeads, kvSeqlen, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMaskOptional = TensorDesc({batch, numHeads, qSeqlen / blockSize, kvSeqlen / blockSize}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto attenMaskOptional = nullptr;
    auto blockShapeOptional = IntArrayDesc({blockSize, blockSize});
    auto actualSeqLengthsOptional = IntArrayDesc({static_cast<int64_t>(qSeqlen), static_cast<int64_t>(qSeqlen)});
    auto actualSeqLengthsKvOptional = IntArrayDesc({static_cast<int64_t>(kvSeqlen), static_cast<int64_t>(kvSeqlen)});
    auto blockTableOptional = nullptr;
    
    char qInputLayout[] = "BNSD";
    char kvInputLayout[] = "BNSD";
    int64_t numKeyValueHeads = kvHeads;
    int64_t maskType = 0;
    double scaleValue = 1.0 / std::sqrt(embeddingSize);
    int64_t innerPrecise = 1;
    int64_t blockSizeAttr = blockSize;
    int64_t preTokens = 65536;
    int64_t nextTokens = 65536;
    int64_t softmaxLseFlag = 0;
    
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto softmaxLseOptional = TensorDesc({batch, numHeads, qSeqlen, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnBlockSparseAttentionGetWorkspaceSize,
                        INPUT(query, key, value, blockSparseMaskOptional, attenMaskOptional, blockShapeOptional,
                              actualSeqLengthsOptional, actualSeqLengthsKvOptional, blockTableOptional,
                              qInputLayout, kvInputLayout, numKeyValueHeads, maskType, scaleValue,
                              innerPrecise, blockSizeAttr, preTokens, nextTokens, softmaxLseFlag),
                        OUTPUT(attentionOut, softmaxLseOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_block_sparse_attention_test, Ascend910B2_block_sparse_attention_layout_mismatch_error)
{
    size_t batch = 2;
    size_t numHeads = 32;
    size_t kvHeads = 8;
    size_t embeddingSize = 128;
    size_t qSeqlen = 1024;
    size_t kvSeqlen = 1024;
    size_t blockSize = 128;
    
    auto query = TensorDesc({qSeqlen * batch, numHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kvSeqlen * batch, kvHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kvSeqlen * batch, kvHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMaskOptional = TensorDesc({batch, numHeads, qSeqlen / blockSize, kvSeqlen / blockSize}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto attenMaskOptional = nullptr;
    auto blockShapeOptional = IntArrayDesc({blockSize, blockSize});
    auto actualSeqLengthsOptional = IntArrayDesc({static_cast<int64_t>(qSeqlen), static_cast<int64_t>(qSeqlen)});
    auto actualSeqLengthsKvOptional = IntArrayDesc({static_cast<int64_t>(kvSeqlen), static_cast<int64_t>(kvSeqlen)});
    auto blockTableOptional = nullptr;
    
    char qInputLayout[] = "TND";
    char kvInputLayout[] = "BNSD";
    int64_t numKeyValueHeads = kvHeads;
    int64_t maskType = 0;
    double scaleValue = 1.0 / std::sqrt(embeddingSize);
    int64_t innerPrecise = 1;
    int64_t blockSizeAttr = blockSize;
    int64_t preTokens = 65536;
    int64_t nextTokens = 65536;
    int64_t softmaxLseFlag = 0;
    
    auto attentionOut = TensorDesc({qSeqlen * batch, numHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto softmaxLseOptional = TensorDesc({qSeqlen * batch, numHeads, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnBlockSparseAttentionGetWorkspaceSize,
                        INPUT(query, key, value, blockSparseMaskOptional, attenMaskOptional, blockShapeOptional,
                              actualSeqLengthsOptional, actualSeqLengthsKvOptional, blockTableOptional,
                              qInputLayout, kvInputLayout, numKeyValueHeads, maskType, scaleValue,
                              innerPrecise, blockSizeAttr, preTokens, nextTokens, softmaxLseFlag),
                        OUTPUT(attentionOut, softmaxLseOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_block_sparse_attention_test, Ascend910B2_block_sparse_attention_bf16)
{
    size_t batch = 2;
    size_t numHeads = 32;
    size_t kvHeads = 8;
    size_t embeddingSize = 128;
    size_t qSeqlen = 1024;
    size_t kvSeqlen = 1024;
    size_t blockSize = 128;
    
    auto query = TensorDesc({qSeqlen * batch, numHeads, embeddingSize}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kvSeqlen * batch, kvHeads, embeddingSize}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kvSeqlen * batch, kvHeads, embeddingSize}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMaskOptional = TensorDesc({batch, numHeads, qSeqlen / blockSize, kvSeqlen / blockSize}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto attenMaskOptional = nullptr;
    auto blockShapeOptional = IntArrayDesc({blockSize, blockSize});
    auto actualSeqLengthsOptional = IntArrayDesc({static_cast<int64_t>(qSeqlen), static_cast<int64_t>(qSeqlen)});
    auto actualSeqLengthsKvOptional = IntArrayDesc({static_cast<int64_t>(kvSeqlen), static_cast<int64_t>(kvSeqlen)});
    auto blockTableOptional = nullptr;
    
    char qInputLayout[] = "TND";
    char kvInputLayout[] = "TND";
    int64_t numKeyValueHeads = kvHeads;
    int64_t maskType = 0;
    double scaleValue = 1.0 / std::sqrt(embeddingSize);
    int64_t innerPrecise = 1;
    int64_t blockSizeAttr = blockSize;
    int64_t preTokens = 65536;
    int64_t nextTokens = 65536;
    int64_t softmaxLseFlag = 0;
    
    auto attentionOut = TensorDesc({qSeqlen * batch, numHeads, embeddingSize}, ACL_BF16, ACL_FORMAT_ND);
    auto softmaxLseOptional = TensorDesc({qSeqlen * batch, numHeads, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnBlockSparseAttentionGetWorkspaceSize,
                        INPUT(query, key, value, blockSparseMaskOptional, attenMaskOptional, blockShapeOptional,
                              actualSeqLengthsOptional, actualSeqLengthsKvOptional, blockTableOptional,
                              qInputLayout, kvInputLayout, numKeyValueHeads, maskType, scaleValue,
                              innerPrecise, blockSizeAttr, preTokens, nextTokens, softmaxLseFlag),
                        OUTPUT(attentionOut, softmaxLseOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_block_sparse_attention_test, Ascend910B2_block_sparse_attention_invalid_inner_precise)
{
    size_t batch = 2;
    size_t numHeads = 32;
    size_t kvHeads = 8;
    size_t embeddingSize = 128;
    size_t qSeqlen = 1024;
    size_t kvSeqlen = 1024;
    size_t blockSize = 128;
    
    auto query = TensorDesc({qSeqlen * batch, numHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kvSeqlen * batch, kvHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kvSeqlen * batch, kvHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMaskOptional = TensorDesc({batch, numHeads, qSeqlen / blockSize, kvSeqlen / blockSize}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto attenMaskOptional = nullptr;
    auto blockShapeOptional = IntArrayDesc({blockSize, blockSize});
    auto actualSeqLengthsOptional = IntArrayDesc({static_cast<int64_t>(qSeqlen), static_cast<int64_t>(qSeqlen)});
    auto actualSeqLengthsKvOptional = IntArrayDesc({static_cast<int64_t>(kvSeqlen), static_cast<int64_t>(kvSeqlen)});
    auto blockTableOptional = nullptr;
    
    char qInputLayout[] = "TND";
    char kvInputLayout[] = "TND";
    int64_t numKeyValueHeads = kvHeads;
    int64_t maskType = 0;
    double scaleValue = 1.0 / std::sqrt(embeddingSize);
    int64_t innerPrecise = 2;
    int64_t blockSizeAttr = blockSize;
    int64_t preTokens = 65536;
    int64_t nextTokens = 65536;
    int64_t softmaxLseFlag = 0;
    
    auto attentionOut = TensorDesc({qSeqlen * batch, numHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto softmaxLseOptional = TensorDesc({qSeqlen * batch, numHeads, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnBlockSparseAttentionGetWorkspaceSize,
                        INPUT(query, key, value, blockSparseMaskOptional, attenMaskOptional, blockShapeOptional,
                              actualSeqLengthsOptional, actualSeqLengthsKvOptional, blockTableOptional,
                              qInputLayout, kvInputLayout, numKeyValueHeads, maskType, scaleValue,
                              innerPrecise, blockSizeAttr, preTokens, nextTokens, softmaxLseFlag),
                        OUTPUT(attentionOut, softmaxLseOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_block_sparse_attention_test, Ascend910B2_block_sparse_attention_invalid_block_shape)
{
    size_t batch = 2;
    size_t numHeads = 32;
    size_t kvHeads = 8;
    size_t embeddingSize = 128;
    size_t qSeqlen = 1024;
    size_t kvSeqlen = 1024;
    size_t blockSize = 128;
    
    auto query = TensorDesc({qSeqlen * batch, numHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kvSeqlen * batch, kvHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kvSeqlen * batch, kvHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMaskOptional = TensorDesc({batch, numHeads, qSeqlen / blockSize, kvSeqlen / blockSize}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto attenMaskOptional = nullptr;
    auto blockShapeOptional = IntArrayDesc({-1, blockSize});
    auto actualSeqLengthsOptional = IntArrayDesc({static_cast<int64_t>(qSeqlen), static_cast<int64_t>(qSeqlen)});
    auto actualSeqLengthsKvOptional = IntArrayDesc({static_cast<int64_t>(kvSeqlen), static_cast<int64_t>(kvSeqlen)});
    auto blockTableOptional = nullptr;
    
    char qInputLayout[] = "TND";
    char kvInputLayout[] = "TND";
    int64_t numKeyValueHeads = kvHeads;
    int64_t maskType = 0;
    double scaleValue = 1.0 / std::sqrt(embeddingSize);
    int64_t innerPrecise = 1;
    int64_t blockSizeAttr = blockSize;
    int64_t preTokens = 65536;
    int64_t nextTokens = 65536;
    int64_t softmaxLseFlag = 0;
    
    auto attentionOut = TensorDesc({qSeqlen * batch, numHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto softmaxLseOptional = TensorDesc({qSeqlen * batch, numHeads, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnBlockSparseAttentionGetWorkspaceSize,
                        INPUT(query, key, value, blockSparseMaskOptional, attenMaskOptional, blockShapeOptional,
                              actualSeqLengthsOptional, actualSeqLengthsKvOptional, blockTableOptional,
                              qInputLayout, kvInputLayout, numKeyValueHeads, maskType, scaleValue,
                              innerPrecise, blockSizeAttr, preTokens, nextTokens, softmaxLseFlag),
                        OUTPUT(attentionOut, softmaxLseOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_block_sparse_attention_test, Ascend910B2_block_sparse_attention_null_query)
{
    size_t batch = 2;
    size_t numHeads = 32;
    size_t kvHeads = 8;
    size_t embeddingSize = 128;
    size_t qSeqlen = 1024;
    size_t kvSeqlen = 1024;
    size_t blockSize = 128;
    
    auto query = nullptr;
    auto key = TensorDesc({kvSeqlen * batch, kvHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kvSeqlen * batch, kvHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMaskOptional = TensorDesc({batch, numHeads, qSeqlen / blockSize, kvSeqlen / blockSize}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto attenMaskOptional = nullptr;
    auto blockShapeOptional = IntArrayDesc({blockSize, blockSize});
    auto actualSeqLengthsOptional = IntArrayDesc({static_cast<int64_t>(qSeqlen), static_cast<int64_t>(qSeqlen)});
    auto actualSeqLengthsKvOptional = IntArrayDesc({static_cast<int64_t>(kvSeqlen), static_cast<int64_t>(kvSeqlen)});
    auto blockTableOptional = nullptr;
    
    char qInputLayout[] = "TND";
    char kvInputLayout[] = "TND";
    int64_t numKeyValueHeads = kvHeads;
    int64_t maskType = 0;
    double scaleValue = 1.0 / std::sqrt(embeddingSize);
    int64_t innerPrecise = 1;
    int64_t blockSizeAttr = blockSize;
    int64_t preTokens = 65536;
    int64_t nextTokens = 65536;
    int64_t softmaxLseFlag = 0;
    
    auto attentionOut = TensorDesc({qSeqlen * batch, numHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto softmaxLseOptional = TensorDesc({qSeqlen * batch, numHeads, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnBlockSparseAttentionGetWorkspaceSize,
                        INPUT(query, key, value, blockSparseMaskOptional, attenMaskOptional, blockShapeOptional,
                              actualSeqLengthsOptional, actualSeqLengthsKvOptional, blockTableOptional,
                              qInputLayout, kvInputLayout, numKeyValueHeads, maskType, scaleValue,
                              innerPrecise, blockSizeAttr, preTokens, nextTokens, softmaxLseFlag),
                        OUTPUT(attentionOut, softmaxLseOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_block_sparse_attention_test, Ascend910B2_block_sparse_attention_with_lse_output)
{
    size_t batch = 2;
    size_t numHeads = 32;
    size_t kvHeads = 8;
    size_t embeddingSize = 128;
    size_t qSeqlen = 1024;
    size_t kvSeqlen = 1024;
    size_t blockSize = 128;
    
    auto query = TensorDesc({qSeqlen * batch, numHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kvSeqlen * batch, kvHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kvSeqlen * batch, kvHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMaskOptional = TensorDesc({batch, numHeads, qSeqlen / blockSize, kvSeqlen / blockSize}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto attenMaskOptional = nullptr;
    auto blockShapeOptional = IntArrayDesc({blockSize, blockSize});
    auto actualSeqLengthsOptional = IntArrayDesc({static_cast<int64_t>(qSeqlen), static_cast<int64_t>(qSeqlen)});
    auto actualSeqLengthsKvOptional = IntArrayDesc({static_cast<int64_t>(kvSeqlen), static_cast<int64_t>(kvSeqlen)});
    auto blockTableOptional = nullptr;
    
    char qInputLayout[] = "TND";
    char kvInputLayout[] = "TND";
    int64_t numKeyValueHeads = kvHeads;
    int64_t maskType = 0;
    double scaleValue = 1.0 / std::sqrt(embeddingSize);
    int64_t innerPrecise = 1;
    int64_t blockSizeAttr = blockSize;
    int64_t preTokens = 65536;
    int64_t nextTokens = 65536;
    int64_t softmaxLseFlag = 1;
    
    auto attentionOut = TensorDesc({qSeqlen * batch, numHeads, embeddingSize}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto softmaxLseOptional = TensorDesc({qSeqlen * batch, numHeads, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnBlockSparseAttentionGetWorkspaceSize,
                        INPUT(query, key, value, blockSparseMaskOptional, attenMaskOptional, blockShapeOptional,
                              actualSeqLengthsOptional, actualSeqLengthsKvOptional, blockTableOptional,
                              qInputLayout, kvInputLayout, numKeyValueHeads, maskType, scaleValue,
                              innerPrecise, blockSizeAttr, preTokens, nextTokens, softmaxLseFlag),
                        OUTPUT(attentionOut, softmaxLseOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}
