/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_lightning_indexer_grad_kl_loss_tiling_general_regbase.cpp
 * \brief
 */

#include "sparse_lightning_indexer_grad_kl_loss_tiling_general_regbase.h"
#include "tiling_base/tiling_templates_registry.h"
#include <tiling/tiling_api.h>

using namespace ge;
using namespace AscendC;

namespace optiling {

bool SparseLightningIndexerGradKLLossTilingBaseRegbase::AnalyzeDimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &queryIndexShape, const gert::Shape &topKShape,
                                                                    size_t layoutLen, const gert::Shape &queryRopeShape, const gert::Shape &keyRopeShape)
{
    // dRopeSize的确定，有queryRopeShape 和 keyRopeShape
    if (layoutLen == 3UL) {
        if (inputLayout[0] == 'T' && inputLayout[1] == 'N' && inputLayout[2] == 'D') {
            int64_t actualSeqQLen = 0;
            int64_t actualSeqKLen = 0;
            int64_t t1Size = queryShape.GetDim(0); // TND只有三个数值 T:0 N:1 D:2
            int64_t t2Size = keyShape.GetDim(0);
            realT1Size = t1Size;
            std::fill(actualSeqLenData.begin(), actualSeqLenData.end(), 0);
            std::fill(actualSeqLenKData.begin(), actualSeqLenKData.end(), 0);
            GetActualSeqLenData(ACTUAL_SEQ_LENGTHS_QUERY_INPUT_INDEX, actualSeqLenData, actualSeqQLen);
            GetActualSeqLenData(ACTUAL_SEQ_LENGTHS_KEY_INPUT_INDEX, actualSeqLenKData, actualSeqKLen);
            OP_CHECK_IF(actualSeqQLen != actualSeqKLen,
                OP_LOGE(opName, "VarLen scene, q[%ld] is not equal k[%ld].", actualSeqQLen, actualSeqKLen), return false);
            // 校验actualQ 对应每一个元素是否大于 actualK ，大于则拦截
            for (int i=0;i<actualSeqQLen;i++) {
                OP_CHECK_IF(actualSeqLenData[i] > actualSeqLenKData[i],
                    OP_LOGE(opName, "Every element of actualSeqLenData must be less than every element of actualSeqLenKData,\
                        but actualSeqLenData[%d]:[%d] is larger than actualSeqLenKData[%d]:[%d].", i, actualSeqLenData[i],
                        i, actualSeqLenKData[i]), return false);
            }
            bSize = actualSeqQLen;
            accumS1 = std::accumulate(actualSeqLenData.begin(), actualSeqLenData.begin() + actualSeqQLen, 0LL);
            accumS2 = std::accumulate(actualSeqLenKData.begin(), actualSeqLenKData.begin() + actualSeqKLen, 0LL);
            OP_CHECK_IF(
                t1Size != accumS1 || t2Size != accumS2,
                OP_LOGE(
                    opName,
                    "Query Tsize(%ld) and key Tsize(%ld) must be equal to sum of seqQLen(%ld) and seqkLen(%ld), respectively.",
                    t1Size, t2Size, accumS1, accumS2),
                return false);
            OP_CHECK_IF(
                s1Size > s2Size || t1Size > t2Size || accumS1 > accumS2,
                OP_LOGE(
                    opName,
                    "Query s1Size(%ld), t1Size(%ld) and the sum of seqQLen(%ld) must be small than Key s2Size(%ld), t2Size(%ld) and seqkLen(%ld), respectively.",
                    s1Size, t1Size, accumS1, s2Size, t2Size, accumS2),
                return false);
            maxS1Val = *std::max_element(actualSeqLenData.begin(), actualSeqLenData.end());
            maxS2Val = *std::max_element(actualSeqLenKData.begin(), actualSeqLenKData.end());
            s1Size = maxS1Val;
            s2Size = maxS2Val;
            n2Size = keyShape.GetDim(1);
            OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
            gSizeQuery = queryShape.GetDim(1) / n2Size;
            gSizeQueryIndex = queryIndexShape.GetDim(1) / n2Size;
            dSizeQuery = queryShape.GetDim(2);
            dSizeQueryIndex = queryIndexShape.GetDim(2);
            kSize = topKShape.GetDim(2);
            OP_CHECK_IF(kSize > BUFFER_SIZE_BYTE_8K || kSize % BUFFER_SIZE_BYTE_1K > 0,
                OP_LOGE(opName, "topK(%d) should be small than 8192, and should be an integer multiple of 1024.", kSize),
                return false);         
            if (hasRope) {
                dQueryRopeSize = queryRopeShape.GetDim(2);
                dKeyRopeSize = keyRopeShape.GetDim(2);
            }
            tilingData->baseParams.set_layoutType(static_cast<uint8_t>(LayoutType::LAYOUT_TND));
            tilingKeyLayout = LayoutType::LAYOUT_TND;
        }
    } else if (layoutLen == 4UL){
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' && inputLayout[3] == 'D') {
            bSize = queryShape.GetDim(0);
            s1Size = queryShape.GetDim(1);
            s2Size = keyShape.GetDim(1);
            n2Size = keyShape.GetDim(2);
            OP_CHECK_IF(
                s1Size > s2Size,
                OP_LOGE(
                    opName,
                    "Query s1Size(%ld) must be small than Key s2Size(%ld).",
                    s1Size, s2Size),
                return false);
            OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
            gSizeQuery = queryShape.GetDim(2) / n2Size;
            gSizeQueryIndex = queryIndexShape.GetDim(2) / n2Size;
            dSizeQuery = queryShape.GetDim(3);
            dSizeQueryIndex = queryIndexShape.GetDim(3);
            kSize = topKShape.GetDim(3);
            OP_CHECK_IF(kSize > BUFFER_SIZE_BYTE_8K || kSize % BUFFER_SIZE_BYTE_1K > 0,
                OP_LOGE(opName, "topK(%d) should be small than 8192, and should be an integer multiple of 1024.", kSize),
                return false);
            if (hasRope) {
                dQueryRopeSize = queryRopeShape.GetDim(3);
                dKeyRopeSize = keyRopeShape.GetDim(3);
            }
            tilingData->baseParams.set_layoutType(static_cast<uint8_t>(LayoutType::LAYOUT_BSND));
            tilingKeyLayout = LayoutType::LAYOUT_BSND;
        }        
    } else {
        return false;
    }

    return true;
}

// 输入shape进行交叉验证，防止数据错误输入
bool SparseLightningIndexerGradKLLossTilingBaseRegbase::CrossShapeVerify(const gert::Shape &queryRopeShape, const gert::Shape &keyRopeShape)
{
    auto queryShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto keyShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto queryIndexShape = context_->GetInputShape(QUERY_INDEX_INPUT_INDEX)->GetStorageShape();
    auto keyIndexShape = context_->GetInputShape(KEY_INDEX_INPUT_INDEX)->GetStorageShape();
    auto weightsShape = context_->GetInputShape(WEIGHT_INPUT_INDEX)->GetStorageShape();
    auto sparseIndicesShape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX)->GetStorageShape();
    auto softmaxMaxShape = context_->GetInputShape(SOFTMAX_MAX_INPUT_INDEX)->GetStorageShape();
    auto softmaxSumShape = context_->GetInputShape(SOFTMAX_SUM_INPUT_INDEX)->GetStorageShape();
    // 下面数字对应shape输入位置
    if (inputLayout[0] == 'T' && inputLayout[1] == 'N' && inputLayout[2] == 'D') {
        int64_t t1Len = queryShape[0];
        int64_t n1Len = queryShape[1];
        int64_t n1indexLen = queryIndexShape[1];
        int64_t t2Len = keyShape[0];
        int64_t n2Len = keyShape[1];
        // 验证T1
        OP_CHECK_IF(queryIndexShape[0] != t1Len || weightsShape[0] != t1Len || softmaxMaxShape[1] != t1Len || softmaxSumShape[1] != t1Len,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify T1 is failed, the value of query[0], query_index[0], weights[0], softmax_max[1] \
                 and softmax_sum[1] are respectively (%ld), (%ld), (%ld), (%ld), (%ld). Their values should be equal.", queryShape[0], queryIndexShape[0], weightsShape[0], softmaxMaxShape[1], softmaxSumShape[1]), return false);
        // 验证N Query数字是否正确
        OP_CHECK_IF(queryShape[1] != NQUERY_SIZE_64 && queryShape[1] != NQUERY_SIZE_128,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape N of query must be one of the {64, 128}, but the value of query[1] is (%ld)", queryShape[1]), return false);        
        // 验证N Index数字是否正确
        OP_CHECK_IF(queryIndexShape[1] != NQUERYINDEX_SIZE_32 && queryIndexShape[1] != NQUERYINDEX_SIZE_64,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape N of query_index must be one of the {32, 64}, but the value of query_index[1] is (%ld).", queryIndexShape[1]), return false);
        OP_CHECK_IF(weightsShape[1] != NQUERYINDEX_SIZE_32 && weightsShape[1] != NQUERYINDEX_SIZE_64,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape N of weights must be one of the {32, 64}, but the value of weights[1] is (%ld).", weightsShape[1]), return false);
        // 验证N Index
        OP_CHECK_IF(queryIndexShape[1] != weightsShape[1],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify N index is failed, the value of query_index[1] and weights[1] are respectively (%ld), (%ld). Their values should be equal.", queryIndexShape[1], weightsShape[1]), return false);
        // 验证T2
        OP_CHECK_IF(keyIndexShape[0] != keyShape[0],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify T2 is failed, the value of key[0] and key_index[0] are respectively (%ld), (%ld). Their values should be equal.", keyShape[0], keyIndexShape[0]), return false);
        // 验证N2 数字是否正确
        OP_CHECK_IF(keyShape[1] != N2_SIZE_1,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of key[1] is (%ld).", keyShape[1]), return false);
        OP_CHECK_IF(keyIndexShape[1] != N2_SIZE_1,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of key_index[1] is (%ld).", keyIndexShape[1]), return false);
        OP_CHECK_IF(softmaxMaxShape[0] != N2_SIZE_1,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of softmax_max[0] is (%ld).", softmaxMaxShape[0]), return false);
        OP_CHECK_IF(softmaxSumShape[0] != N2_SIZE_1,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of softmax_sum[0] is (%ld).", softmaxSumShape[0]), return false);
        // 验证N2
        OP_CHECK_IF(keyIndexShape[1] != keyShape[1] || softmaxMaxShape[0] != keyShape[1] || softmaxSumShape[0] != keyShape[1],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify N2 is failed, the value of key[1], key_index[1], softmax_max[0] and softmax_sum[0] are respectively (%ld), (%ld), (%ld), (%ld). Their values should be equal.",
                 keyShape[1], keyIndexShape[1], softmaxMaxShape[0], softmaxSumShape[0]), return false);
        // 验证D 数字是否正确
        OP_CHECK_IF(queryShape[2] != D_SIZE_512,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of query must be 512, but the value of query[2] is (%ld)", queryShape[2]), return false);
        OP_CHECK_IF(keyShape[2] != D_SIZE_512,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of key must be 512, but the value of key[2] is (%ld)", keyShape[2]), return false);
        OP_CHECK_IF(queryIndexShape[2] != DINDEX_SIZE_128,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of query_index must be 128, but the value of query_index[2] is (%ld)", queryIndexShape[2]), return false);
        OP_CHECK_IF(keyIndexShape[2] != DINDEX_SIZE_128,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of key_index must be 128, but the value of key_index[2] is (%ld)", keyIndexShape[2]), return false);        
        // 验证D
        OP_CHECK_IF(keyShape[2] != queryShape[2],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query-key shape D is failed, the value of query[2] and key[2] are respectively (%ld), (%ld). Their values should be equal.", queryShape[2], keyShape[2]), return false);
        OP_CHECK_IF(queryIndexShape[2] != keyIndexShape[2],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query-key_index shape D is failed, the value of query_index[2] and key_index[2] are respectively (%ld), (%ld). Their values should be equal.", queryIndexShape[2], keyIndexShape[2]), return false);
        // 验证ROPE是否使能
        OP_CHECK_IF(hasRope == 0,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query or key rope is failed, rope can't be null"), return false);
        if (hasRope) {
            // 验证queryrope
            OP_CHECK_IF(queryRopeShape[0] != t1Len || queryRopeShape[1] != n1Len,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query_rope is failed, the value of query_rope[0] is (%ld), it should be equal to tlSize(%ld). \
                    the value of query_rope[1] is (%ld), it should be equal to nQuerySize(%ld).", queryRopeShape[0], t1Len, queryRopeShape[1], n1Len), return false);
            // 验证keyrope
            OP_CHECK_IF(keyRopeShape[0] != t2Len || keyRopeShape[1] != n2Len,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify key_rope is failed, the value of key_rope[0] is (%ld), it should be equal to t2Size(%ld). \
                    the value of key_rope[1] is (%ld), it should be equal to n2Size(%ld)", keyRopeShape[0], t2Len, keyRopeShape[1], n2Len), return false);
            // 验证rope D 数字是否正确
            OP_CHECK_IF(queryRopeShape[2] != DROPE_SIZE_64,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of query_rope must be 64, but the value of query_rope[2] is (%ld).", queryRopeShape[2]), return false);
            OP_CHECK_IF(keyRopeShape[2] != DROPE_SIZE_64,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of key_rope must be 64, but the value of key_rope[2] is (%ld).", keyRopeShape[2]), return false);
            // 验证rope D
            OP_CHECK_IF(queryRopeShape[2] != keyRopeShape[2],
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query_rope D is not equal to key_rope D, the value of query_rope[2] and key_rope[2] is (%ld), (%ld).", queryRopeShape[2], keyRopeShape[2]), return false);
        }           
    } else if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' && inputLayout[3] == 'D') {
        int64_t bLen = queryShape[0];
        int64_t s1Len = queryShape[1];
        int64_t n1Len = queryShape[2];
        int64_t n1indexLen = queryIndexShape[2];
        int64_t s2Len = keyShape[1];
        int64_t n2Len = keyShape[2];
        
        // 验证B
        OP_CHECK_IF(queryIndexShape[0] != bLen || weightsShape[0] != bLen || softmaxMaxShape[0] != bLen || softmaxSumShape[0] != bLen ||
                    keyShape[0] != bLen || keyIndexShape[0] != bLen || sparseIndicesShape[0] != bLen,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify shape B is failed, the value of query[0], key[0], query_index[0], key_index[0], weights[0], softmax_max[0], softmax_sum[0] and sparse_indices[0] are respectively (%ld), (%ld), (%ld), (%ld), (%ld), (%ld), (%ld), (%ld).\
                 Their values should be equal.", queryShape[0], keyShape[0], queryIndexShape[0], keyIndexShape[0], weightsShape[0], softmaxMaxShape[0], softmaxSumShape[0], sparseIndicesShape[0]), return false);
        // 验证s1
        OP_CHECK_IF(queryIndexShape[1] != s1Len || weightsShape[1] != s1Len || softmaxMaxShape[2] != s1Len || softmaxSumShape[2] != s1Len,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify S1 is failed, the value of query[1], query_index[1], weights[1], softmax_max[2] \
                 and softmax_sum[2] are respectively (%ld), (%ld), (%ld), (%ld), (%ld). Their values should be equal.", queryShape[1], queryIndexShape[1], weightsShape[1], softmaxMaxShape[2], softmaxSumShape[2]), return false);
        // 验证s2
        OP_CHECK_IF(keyIndexShape[1] != s2Len,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify shape S2 is failed, the value of key[1] and key_index[1] are respectively (%ld), (%ld). Their values should be equal.", keyShape[1], keyIndexShape[1]), return false);
        // 验证N Query数字是否正确
        OP_CHECK_IF(queryShape[2] != NQUERY_SIZE_64 && queryShape[2] != NQUERY_SIZE_128,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape N of query must be one of the {64, 128}, but the value of query[2] is (%ld)", queryShape[2]), return false);        
        // 验证N Index数字是否正确
        OP_CHECK_IF(queryIndexShape[2] != NQUERYINDEX_SIZE_32 && queryIndexShape[2] != NQUERYINDEX_SIZE_64,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape N of query_index must be one of the {32, 64}, but the value of query_index[2] is (%ld).", queryIndexShape[2]), return false);        
        OP_CHECK_IF(weightsShape[2] != NQUERYINDEX_SIZE_32 && weightsShape[2] != NQUERYINDEX_SIZE_64,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape N of weights must be one of the {32, 64}, but the value of weights[2] is (%ld).", weightsShape[2]), return false);         
        // 验证N Index
        OP_CHECK_IF(queryIndexShape[2] != weightsShape[2],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify N index is failed, the value of query_index[2] and weights[2] are respectively (%ld), (%ld). Their values should be equal.", queryIndexShape[2], weightsShape[2]), return false);
        // 验证N2 数字是否正确
        OP_CHECK_IF(keyShape[2] != N2_SIZE_1,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of key[2] is (%ld).", keyShape[2]), return false);        
        OP_CHECK_IF(keyIndexShape[2] != N2_SIZE_1,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of key_index[2] is (%ld).", keyIndexShape[2]), return false);
        OP_CHECK_IF(softmaxMaxShape[1] != N2_SIZE_1,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of softmax_max[1] is (%ld).", softmaxMaxShape[1]), return false);        
        OP_CHECK_IF(softmaxSumShape[1] != N2_SIZE_1,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, N2 must be 1, but the value of softmax_sum[1] is (%ld).", softmaxSumShape[1]), return false);
        // 验证N2
        OP_CHECK_IF(keyIndexShape[2] != n2Len || softmaxMaxShape[1] != n2Len || softmaxSumShape[1] != n2Len,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify N2 is failed, the value of key[2], key_index[2], softmax_max[1] and softmax_sum[1] are respectively (%ld), (%ld), (%ld), (%ld). Their values should be equal.",
                 keyShape[2], keyIndexShape[2], softmaxMaxShape[1], softmaxSumShape[1]), return false);
        // 验证D 数字是否正确
        OP_CHECK_IF(queryShape[3] != D_SIZE_512,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of query must be 512, but the value of query[3] is (%ld)", queryShape[3]), return false);
        OP_CHECK_IF(keyShape[3] != D_SIZE_512,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of key must be 512, but the value of key[3] is (%ld)", keyShape[3]), return false);
        OP_CHECK_IF(queryIndexShape[3] != DINDEX_SIZE_128,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of query_index must be 128, but the value of query_index[3] is (%ld)", queryIndexShape[3]), return false);        
        OP_CHECK_IF(keyIndexShape[3] != DINDEX_SIZE_128,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of key_index must be 128, but the value of key_index[3] is (%ld)", keyIndexShape[3]), return false);       
        // 验证D
        OP_CHECK_IF(keyShape[3] != queryShape[3],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query-key shape D is failed, the value of query[3] and key[3] are respectively (%ld), (%ld). Their values should be equal.", queryShape[3], keyShape[3]), return false);
        OP_CHECK_IF(queryIndexShape[3] != keyIndexShape[3],
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query-key_index shape D is Failed, the value of query_index[3] and key_index[3] are respectively (%ld), (%ld). Their values should be equal.", queryIndexShape[3], keyIndexShape[3]), return false);
        // 验证ROPE是否使能
        OP_CHECK_IF(hasRope == 0,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query or key rope is failed, rope can't be null"), return false);
        if (hasRope) {
            // 验证queryrope
            OP_CHECK_IF(queryRopeShape[0] != bLen || queryRopeShape[1] != s1Len || queryRopeShape[2] != n1Len,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query_rope is failed, the value of query_rope[0] is (%ld), it should be equal to bSize(%ld). \
                    the value of query_rope[1] is (%ld), it should be equal to s1Size(%ld), the value of query_rope[2] is (%ld), it should be equal to nQuerySize(%ld).", queryRopeShape[0], bLen, queryRopeShape[1], s1Len, queryRopeShape[2], n1Len), return false);
            // 验证keyrope
            OP_CHECK_IF(keyRopeShape[0] != bLen || keyRopeShape[1] != s2Len || keyRopeShape[2] != n2Len,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify key_rope is failed, the value of key_rope[0] is (%ld), it should be equal to bSize(%ld), the value of key_rope[1] is (%ld), it should be equal to s2Size(%ld). \
                    the value of key_rope[2] is (%ld), it should be equal to n2Size(%ld)", keyRopeShape[0], bLen, keyRopeShape[1], s2Len, keyRopeShape[2], n2Len), return false);
            // 验证rope D 数字是否正确
            OP_CHECK_IF(queryRopeShape[3] != DROPE_SIZE_64,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of query_rope  must be 64, but the value of query_rope[3] is (%ld).", queryRopeShape[3]), return false);
            OP_CHECK_IF(keyRopeShape[3] != DROPE_SIZE_64,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify failed, shape D of key_rope must be 64, but the value of key_rope[3] is (%ld).", keyRopeShape[3]), return false);           
            // 验证rope D
            OP_CHECK_IF(queryRopeShape[3] != keyRopeShape[3],
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify query_rope D is not equal to key_rope D, the value of queryRope[3] and keyRope[3] is (%ld), (%ld).", queryRopeShape[3], keyRopeShape[3]), return false);
        }
    }
    return true;
}

bool SparseLightningIndexerGradKLLossTilingBaseRegbase::AnalyzeLayout()
{
    auto &queryShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto &keyShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto &queryIndexShape = context_->GetInputShape(QUERY_INDEX_INPUT_INDEX)->GetStorageShape();
    auto &keyIndexShape = context_->GetInputShape(KEY_INDEX_INPUT_INDEX)->GetStorageShape();
    auto &topKShape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX)->GetStorageShape();

    auto queryRope = context_->GetOptionalInputShape(QUERY_ROPE_INPUT_INDEX); 
    bool hasQueryRope = queryRope != nullptr && queryRope->GetStorageShape().GetDimNum() != 0;
    auto keyRope = context_->GetOptionalInputShape(KEY_ROPE_INPUT_INDEX);
    bool hasKeyRope = keyRope != nullptr && keyRope->GetStorageShape().GetDimNum() != 0;
    if (hasQueryRope ^ hasKeyRope) {
        OP_LOGE(opName, "query_rope and key_rope should be present or absent at the same time, check this.");
        return false;
    }
    // TND和BSND的dim位置不同来赋值
    hasRope = hasQueryRope && hasKeyRope;
    auto &queryRopeShape = queryRope->GetStorageShape();
    auto &keyRopeShape = keyRope->GetStorageShape();

    size_t layoutLen = strlen(inputLayout);
    OP_CHECK_IF(queryShape.GetDimNum() != layoutLen || keyShape.GetDimNum() != layoutLen ||
        queryIndexShape.GetDimNum() != layoutLen || keyIndexShape.GetDimNum() != layoutLen, OP_LOGE(opName, "Invalid layout[%s].", inputLayout), return false);
    OP_CHECK_IF(!CrossShapeVerify(queryRopeShape, keyRopeShape), OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify Failed"), return false);    
    OP_CHECK_IF(!AnalyzeDimLayout(queryShape, keyShape, queryIndexShape, topKShape, layoutLen, queryRopeShape, keyRopeShape),
               OP_LOGE(opName, "Layout: %s, Run Failed", inputLayout), return false);
    OP_CHECK_IF(gSizeQuery == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "gSizeQuery is zero"), return false);
    OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "n2Size is zero"), return false);
    OP_CHECK_IF(dSizeQuery <= 0,
                 OPS_REPORT_VECTOR_INNER_ERR(opName, "dSizeQuery  is not support <= 0"), return false);
    return true;
}

ge::graphStatus SparseLightningIndexerGradKLLossTilingBaseRegbase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "TilingContext: %s.", GetTilingContextDebugStr().c_str());

    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(opName, "invalid context."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout(),
               OP_LOGE(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);
    // 将基本输入参数传给tilingdata
    sliGradkllossBaseParams_->set_bSize(bSize);
    sliGradkllossBaseParams_->set_n2Size(n2Size);
    sliGradkllossBaseParams_->set_gSizeQuery(gSizeQuery);
    sliGradkllossBaseParams_->set_gSizeQueryIndex(gSizeQueryIndex);
    sliGradkllossBaseParams_->set_s1Size(s1Size);
    sliGradkllossBaseParams_->set_s2Size(s2Size);
    sliGradkllossBaseParams_->set_dSizeQuery(dSizeQuery);
    sliGradkllossBaseParams_->set_dSizeQueryIndex(dSizeQueryIndex);
    sliGradkllossBaseParams_->set_kSize(kSize);
    sliGradkllossBaseParams_->set_sparseMode(sparseMode);
    sliGradkllossBaseParams_->set_scaleValue(scaleValue);
    
    OP_LOGW(context_, "INPUTPARAM bsize:[%d], n2Size:[%d], gSizeQuery:[%d], dSizeQueryIndex:[%d], s1Size:[%d], s2Size:[%d], dSizeQuery:[%d], dSizeQueryIndex:[%d], kSize:[%d], sparseMode:[%d], scaleValue:[%f] .", 
            bSize, n2Size, gSizeQuery, gSizeQueryIndex, s1Size, s2Size, dSizeQuery, dSizeQueryIndex, kSize, sparseMode, scaleValue);
    return ge::GRAPH_SUCCESS;
}

void SparseLightningIndexerGradKLLossTilingBaseRegbase::SetSparseParamsRegbase()
{
    int64_t validAicNum = static_cast<int64_t>(sliGradkllossMultiCoreParams_->get_coreNum());
    int64_t totalSize = sliGradkllossMultiCoreParams_->get_totalSize();
    int64_t *sparseStartIdx = sliGradkllossMultiCoreParams_->get_bS1Ptr();
    int64_t splitFactorSize = sliGradkllossMultiCoreParams_->get_splitFactorSize();
    std::vector<int64_t> sparseValidArray(totalSize, 0);
    InitSparseValidArray(sparseValidArray, sparseMode);
    SetSparseStartIdx(sparseValidArray, validAicNum, totalSize, sparseStartIdx, splitFactorSize, static_cast<int64_t>(MAX_CORE_NUM_REGBASE));
}

void SparseLightningIndexerGradKLLossTilingBaseRegbase::SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum)
{
    int64_t actualUsedCoreNum = std::min(totalSize, static_cast<int64_t>(coreNum));
    sliGradkllossMultiCoreParams_->set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    sliGradkllossMultiCoreParams_->set_totalSize(totalSize);
    sliGradkllossMultiCoreParams_->set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
}

ge::graphStatus SparseLightningIndexerGradKLLossTilingBaseRegbase::DoOpTiling()
{
    OP_LOGD(context_, "try template[%s]", templateName);
    // 无多余操作，分核，目前只实现TND场景分核
    int64_t totalSize = CalcTotalSize();
    SetMultiCoreParamsRegbase(totalSize, static_cast<int64_t>(aicNum));
    context_->SetBlockDim(sliGradkllossMultiCoreParams_->get_coreNum()); // 使用的核数确定

    std::vector<int64_t> shapeVec = {1, kSize};
    ge::Shape srcShape(shapeVec);
    int64_t softmaxTmpBufferSize = BUFFER_SIZE_BYTE_32K; // 需要32KB
    if (kSize > BUFFER_SIZE_BYTE_2K) {
        softmaxTmpBufferSize = BUFFER_SIZE_BYTE_33K; //kSize >2048 需要33kB
    }
    SoftMaxTilingFunc(srcShape, 4, softmaxTmpBufferSize, tilingData->vectorParams.softmaxYTilingData);

    SetSparseParamsRegbase();
    int64_t totalsize = InitOutputSplit(); // output分核
    int64_t singlecoresize = static_cast<uint32_t>(CeilDivision(totalsize, static_cast<int64_t>(aivNum))); // 输出k-index总大小TD或者BSD 除以 总的aiv核数 向上取整
    initoutput->set_singleCoreSize(singlecoresize);
    initoutput->set_totalOutputSize(totalsize);
    OP_LOGD(context_, "ending template[%s]", templateName);
    return ge::GRAPH_SUCCESS;
}

uint64_t SparseLightningIndexerGradKLLossTilingBaseRegbase::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint8_t>(hasRope), static_cast<uint8_t>(tilingKeyLayout), 
        static_cast<uint8_t>(tilingKeyLayout), static_cast<uint8_t>(sparseMode), static_cast<uint8_t>(deterministic));
}

ge::graphStatus SparseLightningIndexerGradKLLossTilingBaseRegbase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    int64_t reduceSumOffset = PING_PONG_VALUE * kSize * sizeof(float);
    int64_t reluOffset = gSizeQueryIndex * kSize * sizeof(float); 
    int64_t gatherSYOffset = kSize * dSizeQueryIndex * 2; // 2代表输入D类型大小
    int64_t scatterAddOutSize;
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        scatterAddOutSize = accumS2 * dSizeQueryIndex * sizeof(float); //batch
    } else {
        scatterAddOutSize = bSize * s2Size * dSizeQueryIndex * sizeof(float); //batch
    }

    int64_t singlecoreTotalSize = reduceSumOffset + reluOffset + gatherSYOffset;
    int64_t multicoreTotalsize = singlecoreTotalSize * static_cast<int64_t>(sliGradkllossMultiCoreParams_->get_coreNum()) + scatterAddOutSize;
    workspaces[0] = static_cast<size_t>(multicoreTotalsize) + WORK_SPACE_RESERVE_SIZE; // 预留16M空间必须加;
    OP_LOGW(context_, "workspace size:[%ld], multicoreTotalsize:[%ld]", workspaces[0], multicoreTotalsize);

    sliGradkllossWorkSpaceOffsetParams_->set_reduceSumOffset(reduceSumOffset);
    sliGradkllossWorkSpaceOffsetParams_->set_reluOffset(reluOffset);
    sliGradkllossWorkSpaceOffsetParams_->set_gatherSYOffset(gatherSYOffset);
    sliGradkllossWorkSpaceOffsetParams_->set_scatterAddOutSize(scatterAddOutSize);
    sliGradkllossWorkSpaceOffsetParams_->set_singlecoreTotalSize(singlecoreTotalSize);
    sliGradkllossWorkSpaceOffsetParams_->set_multicoreTotalsize(multicoreTotalsize);

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(SparseLightningIndexerGradKLLoss, SparseLightningIndexerGradKLLossTilingBaseRegbase, static_cast<int32_t>(NpuArch::DAV_3510), 1);
} // namespace optiling
