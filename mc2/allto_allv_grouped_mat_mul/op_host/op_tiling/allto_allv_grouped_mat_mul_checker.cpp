/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file allto_allv_grouped_mat_mul_checker.cpp
 * \brief
 */
#include "allto_allv_grouped_mat_mul_checker.h"

namespace optiling {
void AlltoAllvGmmChecker::CheckerSetMNK(int32_t maxM,int32_t maxN,int32_t maxK,int32_t maxMForMM,int32_t maxNForMM,int32_t maxKForMM)
{
    maxM_ = maxM;
    maxN_ = maxN;
    maxK_ = maxK;
    maxMForMM_ = maxMForMM;
    maxNForMM_ = maxNForMM;
    maxKForMM_ = maxKForMM;
}

ge::graphStatus AlltoAllvGmmChecker::CheckMKN(const gert::TilingContext *context)
{
    // maxM_ = tilingData->commonTilingInfo.A;
    // maxK_ = tilingData->commonTilingInfo.H1;
    // maxN_ = tilingData->commonTilingInfo.N1;
    // (void)context;  // Unused
    // OP_TILING_CHECK(mmDataTypeSize == 0,
    //     OP_LOGE(A_INNER_DEBUG,
    //         "GMM get matmul dtype[%s] size is 0.",
    //         ge::TypeUtils::DataTypeToAscendString(mmDType_).GetString()),
    //     return ge::GRAPH_FAILED);
    // uint32_t numInOneBlk = AscendC::ONE_BLK_SIZE / mmDataTypeSize;
    // OP_TILING_CHECK(numInOneBlk == 0, OP_LOGE(A_INNER_DEBUG, "GMM numInOneBlk cannot be 0."), return ge::GRAPH_FAILED);
    // int64_t maxMKN = INT_MAX / numInOneBlk * numInOneBlk;
    // OP_TILING_CHECK(maxM_ > maxMKN || maxN_ > maxMKN || maxK_ > maxMKN,
    //     OP_LOGE(A_INNER_DEBUG, "32B-aligned m, n or k axis for gmm is out of range int32!"),
    //     return ge::GRAPH_FAILED);
    // if (tilingData->commonTilingInfo.isNeedMM) {
    //     OP_TILING_CHECK(maxMForMM_ > maxMKN || maxNForMM_ > maxMKN || maxKForMM_ > maxMKN,
    //         OP_LOGE(A_INNER_DEBUG, "32B-aligned m, n or k axis for mm is out of range int32!"),
    //         return ge::GRAPH_FAILED);
    // }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmChecker::CheckSendRecvDataVolumn(const gert::TilingContext *context) const
{
    // // Data for communication between cards [2M,100M]
    // uint64_t eExpert = tilingData->commonTilingInfo.E_ep;
    // uint64_t epWorldSize = tilingData->commonTilingInfo.epWorldSize;
    // uint64_t recvSendMin = static_cast<uint64_t>(2U * 1024U * 1024U);  // Lower traffic threshold 2MB=2*1024*1024

    // auto attrs = context->GetAttrs();
    // OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);

    // auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    // auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    // OP_TILING_CHECK((sendCountsPtr == nullptr) || (recvCountsPtr == nullptr),
    //     OP_LOGE(A_INNER_DEBUG, "sendCountsPtr or recvCountsPtr is null."),
    //     return ge::GRAPH_FAILED);

    // const uint64_t *sendCounts = static_cast<const uint64_t *>(sendCountsPtr->GetData());
    // const uint64_t *recvCounts = static_cast<const uint64_t *>(recvCountsPtr->GetData());
    // uint64_t recvSum = 0U;
    // uint64_t sendSum = 0U;
    // uint64_t H1 = tilingData->commonTilingInfo.H1;
    // uint64_t bsk = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(0);
    // uint64_t a = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(0);
    // auto platformInfo = context->GetPlatformInfo();
    // platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    // if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_93) {
    //     for (uint64_t i = 1U; i <= epWorldSize; i++) {
    //         recvSum = 0U;
    //         sendSum = 0U;
    //         for (uint64_t j = (i - 1U) * eExpert; j <= i * eExpert - 1U; j++) {
    //             OP_TILING_CHECK((sendCounts[j] < NUM_ZERO) || (sendCounts[j] > bsk),
    //                 OP_LOGE(A_INNER_DEBUG,
    //                     "sendCounts[%lu] should be in [0, bsK[%lu]], but get %lu",
    //                     j,
    //                     bsk,
    //                     sendCounts[j]),
    //                 return ge::GRAPH_FAILED);
    //             OP_TILING_CHECK((recvCounts[j] < NUM_ZERO) || (recvCounts[j] > a),
    //                 OP_LOGE(
    //                     A_INNER_DEBUG, "recvCounts[%lu] should be in [0, a[%lu]], but get %lu", j, a, recvCounts[j]),
    //                 return ge::GRAPH_FAILED);
    //             recvSum += recvCounts[j] * H1 * 2U;
    //             sendSum += sendCounts[j] * H1 * 2U;  // /sizeof(gmmX) = 2U
    //         }
    //         OP_TILING_CHECK(recvSum < recvSendMin,
    //             OP_LOGE(A_INNER_DEBUG,
    //                 "rank %lu:sum(recvCounts[%lu, %lu]) * H1 * sizeof dtype(gmmx) should be greater than or equal to "
    //                 "2MB,"
    //                 "but got %lu Byte!",
    //                 i - 1U,
    //                 (i - 1U) * eExpert,
    //                 i * eExpert - 1U,
    //                 recvSum),
    //             return ge::GRAPH_FAILED);
    //         OP_TILING_CHECK(sendSum < recvSendMin,
    //             OP_LOGE(A_INNER_DEBUG,
    //                 "rank %lu:sum(sendCounts[%lu, %lu]) * H1 * sizeof dtype(gmmx) should be greater than or equal to "
    //                 "2MB,"
    //                 "but got %lu Byte!",
    //                 i - 1U,
    //                 (i - 1U) * eExpert,
    //                 i * eExpert - 1U,
    //                 sendSum),
    //             return ge::GRAPH_FAILED);
    //     }
    // }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmChecker::CheckShapeSize(const gert::TilingContext *context) const
{
    // OP_TILING_CHECK(
    //     (context->GetInputShape(GMM_X_INDEX) == nullptr) || (context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr),
    //     OP_LOGE(A_INNER_DEBUG, "GetInputShape gmmX or gmmWeight returned null."),
    //     return ge::GRAPH_FAILED);

    // uint64_t BSK = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(0);
    // if (BSK <= NUM_ZERO || BSK >= MAX_BSK) {
    //     OP_LOGE(A_INNER_DEBUG, "BSK should be in (0, 52428800), but got %lu!", BSK);
    //     return ge::GRAPH_FAILED;
    // }
    // uint64_t H1 = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(1);
    // OP_TILING_CHECK((H1 <= NUM_ZERO) || (H1 >= MAX_SHAPE_SIZE),
    //     OP_LOGE(A_INNER_DEBUG, "H1 should be in (0, 65536), but got %lu!", H1),
    //     return ge::GRAPH_FAILED);

    // uint64_t N1 = tilingData->commonTilingInfo.isGmmWeightTrans
    //                   ? context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(1)
    //                   : context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(NUM_TWO);
    // OP_TILING_CHECK(N1 <= NUM_ZERO || N1 >= MAX_SHAPE_SIZE,
    //     OP_LOGE(A_INNER_DEBUG, "N1 should be in (0, 65536), but got %lu!", N1),
    //     return ge::GRAPH_FAILED);

    // if (tilingData->commonTilingInfo.isNeedMM) {
    //     uint64_t BS = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(0);
    //     if (BS <= NUM_ZERO) {
    //         OP_LOGE(A_INNER_DEBUG, "BS should be larger than 0, but got %lu!", BS);
    //         return ge::GRAPH_FAILED;
    //     }
    //     uint64_t H2 = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(1);
    //     if (H2 <= NUM_ZERO || H2 > MAX_SHARED_H_SHAPE_SIZE) {
    //         OP_LOGE(A_INNER_DEBUG, "H2 should be in (0, 12288], but got %lu!", H2);
    //         return ge::GRAPH_FAILED;
    //     }
    //     uint64_t N2 = tilingData->commonTilingInfo.isMmWeightTrans
    //                       ? context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(0)
    //                       : context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(1);
    //     if (N2 <= NUM_ZERO || N2 >= MAX_SHAPE_SIZE) {
    //         OP_LOGE(A_INNER_DEBUG, "N2 should be in (0, 65536), but got %lu!", N2);
    //         return ge::GRAPH_FAILED;
    //     }
    //     uint64_t topK = BSK / BS;
    //     if (topK < NUM_TWO || topK > NUM_EIGHT) {
    //         OP_LOGE(A_INNER_DEBUG, "topK should be in [2, 8], but got %lu!", topK);
    //         return ge::GRAPH_FAILED;
    //     }
    // }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmChecker::CheckAttrsShapeSize(const gert::TilingContext *context) const
{
    // uint64_t E_ep = tilingData->commonTilingInfo.E_ep;
    // if (E_ep <= NUM_ZERO || E_ep > NUM_THIRTYTWO) {
    //     OP_LOGE(A_INNER_DEBUG, "E_ep should be in (0, 32], but got %lu!", E_ep);
    //     return ge::GRAPH_FAILED;
    // }
    // uint64_t epWorldSize = tilingData->commonTilingInfo.epWorldSize;
    // auto platformInfo = context->GetPlatformInfo();
    // platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    // std::vector<int64_t> epWorldSizeOptional;
    // std::string epWorldSizeNum;
    // if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
    //     epWorldSizeOptional = {2, 4, 8, 16, 32, 64};  // A5 limits the epWorldSize to {248163264}
    // } else {
    //     epWorldSizeOptional = {8, 16, 32, 64, 128};  // A3 limits the epWorldSize to{8163264, 128}
    // }
    // for (size_t i = 0; i < epWorldSizeOptional.size(); i++) {
    //     epWorldSizeNum += (std::to_string(epWorldSizeOptional[i]) + " ");
    // }
    // OP_TILING_CHECK(
    //     std::find(epWorldSizeOptional.begin(), epWorldSizeOptional.end(), epWorldSize) == epWorldSizeOptional.end(),
    //     OP_LOGE(A_INNER_DEBUG, "epWorldSize[%lu] should be %s!", epWorldSize, epWorldSizeNum.c_str()),
    //     return ge::GRAPH_FAILED);
    // auto attrs = context->GetAttrs();
    // OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);
    // auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    // auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    // OP_TILING_CHECK((sendCountsPtr == nullptr) || (recvCountsPtr == nullptr),
    //     OP_LOGE(A_INNER_DEBUG, "sendCountsPtr or recvCountsPtr is null."),
    //     return ge::GRAPH_FAILED);
    // uint64_t sendCountsSize = sendCountsPtr->GetSize();
    // uint64_t recvCountsSize = recvCountsPtr->GetSize();
    // OP_TILING_CHECK(sendCountsSize != recvCountsSize,
    //     OP_LOGE(A_INNER_DEBUG,
    //         "The size of sendCounts(e*ep) %lu should be equal to recvCounts(e*ep) %lu !",
    //         sendCountsSize,
    //         recvCountsSize),
    //     return ge::GRAPH_FAILED);
    // if (E_ep * epWorldSize != sendCountsSize) {
    //     OP_LOGE(A_INNER_DEBUG,
    //         "The first dim of gmmWeight(e, H1, N1) %lu  multi epWorldSize %lu shoubl be equal to the size of "
    //         "sendCounts(e*ep) %lu!",
    //         E_ep,
    //         epWorldSize,
    //         sendCountsSize);
    //     return ge::GRAPH_FAILED;
    // }
    // if ((E_ep * epWorldSize <= NUM_ZERO) || (E_ep * epWorldSize > MAX_EXPERT_NUM)) {
    //     OP_LOGE(A_INNER_DEBUG,
    //         "The size of send_counts(e*ep) and recv_counts(e*ep) should be in (0, 256], but got %lu!",
    //         E_ep * epWorldSize);
    //     return ge::GRAPH_FAILED;
    // }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmChecker::CheckAttrsShapeRelation(const gert::TilingContext *context) const
{
    // auto attrs = context->GetAttrs();
    // OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);

    // auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    // auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    // OP_TILING_CHECK((sendCountsPtr == nullptr) || (recvCountsPtr == nullptr),
    //     OP_LOGE(A_INNER_DEBUG, "sendCountsPtr or recvCountsPtr is null."),
    //     return ge::GRAPH_FAILED);

    // uint64_t sendCountsSize = sendCountsPtr->GetSize();
    // uint64_t recvCountsSize = recvCountsPtr->GetSize();

    // errno_t ret = memcpy_s(&(tilingData->aicpuTiling.sendCnt),
    //     MAX_EXPERT_NUM * sizeof(int64_t),
    //     sendCountsPtr->GetData(),
    //     sendCountsPtr->GetSize() * sizeof(int64_t));
    // if (ret != EOK) {
    //     OP_LOGE(A_INNER_DEBUG, "memcpy_s failed, ret = %d.", ret);
    //     return ge::GRAPH_FAILED;
    // }
    // ret = memcpy_s(&(tilingData->aicpuTiling.recvCnt),
    //     MAX_EXPERT_NUM * sizeof(int64_t),
    //     recvCountsPtr->GetData(),
    //     recvCountsPtr->GetSize() * sizeof(int64_t));
    // if (ret != EOK) {
    //     OP_LOGE(A_INNER_DEBUG, "memcpy_s failed, ret = %d.", ret);
    //     return ge::GRAPH_FAILED;
    // }

    // const uint64_t *sendCounts = static_cast<const uint64_t *>(sendCountsPtr->GetData());
    // uint64_t sendCountsSum = std::accumulate(sendCounts, sendCounts + sendCountsSize, 0ULL);
    // OP_TILING_CHECK(sendCountsSum != tilingData->commonTilingInfo.BSK,
    //     OP_LOGE(A_INNER_DEBUG,
    //         "The sum of sendCounts %lu should be equal to BSK %lu!",
    //         sendCountsSum,
    //         tilingData->commonTilingInfo.BSK),
    //     return ge::GRAPH_FAILED);

    // const uint64_t *recvCounts = static_cast<const uint64_t *>(recvCountsPtr->GetData());
    // uint64_t recvCountsSum = std::accumulate(recvCounts, recvCounts + recvCountsSize, 0ULL);
    // OP_TILING_CHECK(recvCountsSum != tilingData->commonTilingInfo.A,
    //     OP_LOGE(A_INNER_DEBUG,
    //         "The sum of recvCounts %lu should be equal to A %lu!",
    //         recvCountsSum,
    //         tilingData->commonTilingInfo.A),
    //     return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmChecker::CheckShapeRelation(const gert::TilingContext *context) const
{
    // OP_TILING_CHECK(
    //     (context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr) || (context->GetInputShape(GMM_X_INDEX) == nullptr),
    //     OP_LOGE(A_INNER_DEBUG, "GetInputShape gmmX or gmmWeight returned nullptr."),
    //     return ge::GRAPH_FAILED);

    // uint64_t gmmWeightH1 = tilingData->commonTilingInfo.isGmmWeightTrans
    //                            ? context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(NUM_TWO)
    //                            : context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(1);
    // uint64_t gmmXH1 = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(1);
    // OP_TILING_CHECK(gmmXH1 != gmmWeightH1,
    //     OP_LOGE(A_INNER_DEBUG,
    //         "The H1 %lu of gmmX(BSK, H1) should be equal to the H1 %lu of gmmWeight(e, H1, N1) !",
    //         gmmXH1,
    //         gmmWeightH1),
    //     return ge::GRAPH_FAILED);

    // if (tilingData->commonTilingInfo.isNeedMM) {
    //     uint64_t mmXH2 = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(1);
    //     uint64_t mmWeightH2 = tilingData->commonTilingInfo.isMmWeightTrans
    //                               ? context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(1)
    //                               : context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(0);
    //     OP_TILING_CHECK(mmXH2 != mmWeightH2,
    //         OP_LOGE(A_INNER_DEBUG,
    //             "The H2 %lu of mmX(BS, H2) should be equal to the H2 %lu of mmWeight(H2, N2)!",
    //             mmXH2,
    //             mmWeightH2),
    //         return ge::GRAPH_FAILED);

    //     uint64_t mmXBS = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(0);
    //     uint64_t mmYBS = context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDim(0);
    //     OP_TILING_CHECK(mmXBS != mmYBS,
    //         OP_LOGE(
    //             A_INNER_DEBUG, "The BS %lu of mmX(BS, H2) should be equal to the BS %lu of mmY(BS, N2)!", mmXBS, mmYBS),
    //         return ge::GRAPH_FAILED);
    // }

    // if (tilingData->commonTilingInfo.isPermuteOut) {
    //     OP_TILING_CHECK((context->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr) ||
    //                         (context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) == nullptr),
    //         OP_LOGE(A_INNER_DEBUG, "GetPermuteOutputShape GmmY or permuteOut returned null."),
    //         return ge::GRAPH_FAILED);
    //     uint64_t gmmYA = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(0);
    //     uint64_t permuteA = context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDim(0);
    //     uint64_t permuteH1 = context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDim(1);
    //     OP_TILING_CHECK(gmmXH1 != permuteH1,
    //         OP_LOGE(A_INNER_DEBUG,
    //             "The H1 %lu of gmmX(BSK, H1) should be equal to the H1 %lu of permuteOut(A, H1)!",
    //             gmmXH1,
    //             permuteH1),
    //         return ge::GRAPH_FAILED);
    //     OP_TILING_CHECK(gmmYA != permuteA,
    //         OP_LOGE(A_INNER_DEBUG,
    //             "The A %lu of gmmY(A, H1) should be equal to the A %lu of permuteOut(A, H1)!",
    //             gmmYA,
    //             permuteA),
    //         return ge::GRAPH_FAILED);
    // }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmChecker::CheckMmShapeDims(const gert::TilingContext *context) const
{
    // if (tilingData->commonTilingInfo.isNeedMM) {
    //     if (context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
    //         OP_LOGE(A_INNER_DEBUG,
    //             "The dim of mmX(BS, H2) should be 2, but got %lu!",
    //             context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDimNum());
    //         return ge::GRAPH_FAILED;
    //     }
    //     if (context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
    //         OP_LOGE(A_INNER_DEBUG,
    //             "The dim of mmWeight(H2, N2) should be 2, but got %lu!",
    //             context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDimNum());
    //         return ge::GRAPH_FAILED;
    //     }
    //     if (context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
    //         OP_LOGE(A_INNER_DEBUG,
    //             "The dim of mmY(BS, N2) should be 2, but got %lu!",
    //             context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum());
    //         return ge::GRAPH_FAILED;
    //     }
    // } else {
    //     OP_TILING_CHECK((context->GetOutputShape(OUTPUT_MM_Y_INDEX) != nullptr &&
    //                         context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum() != NUM_ZERO),
    //         OP_LOGE(A_INNER_DEBUG, "The mmY should be null when mmX and mmWeight are null!"),
    //         return ge::GRAPH_FAILED);
    //     if (tilingData->commonTilingInfo.isMmWeightTrans) {
    //         OP_LOGE(A_INNER_DEBUG, "The trans_mm_weight should be false when mmX mmWeight mmY is null!");
    //         return ge::GRAPH_FAILED;
    //     }
    // }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmChecker::CheckShapeDims(const gert::TilingContext *context)
{
    // OP_TILING_CHECK(
    //     (context->GetInputShape(GMM_X_INDEX) == nullptr) || (context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr),
    //     OP_LOGE(A_INNER_DEBUG, "GetInputShape gmmX or gmmWeight returned null."),
    //     return ge::GRAPH_FAILED);
    // OP_TILING_CHECK((context->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr),
    //     OP_LOGE(A_INNER_DEBUG, "GetOutputShape gmmY returned null."),
    //     return ge::GRAPH_FAILED);

    // if (context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
    //     OP_LOGE(A_INNER_DEBUG,
    //         "The dim of gmmX(BSK, H1) should be 2, but got %lu!",
    //         context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDimNum());
    //     return ge::GRAPH_FAILED;
    // }
    // if (context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDimNum() != NUM_THREE) {
    //     OP_LOGE(A_INNER_DEBUG,
    //         "The dim of gmmWeight(e, H1, N1) should be 3, but got %lu!",
    //         context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDimNum());
    //     return ge::GRAPH_FAILED;
    // }
    // if (context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
    //     OP_LOGE(A_INNER_DEBUG,
    //         "The dim of gmmY(A, N1) should be 2, but got %lu!",
    //         context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDimNum());
    //     return ge::GRAPH_FAILED;
    // }
    // if (tilingData->commonTilingInfo.isNeedMM) {
    //     OP_TILING_CHECK(context->GetOptionalInputShape(MM_WEIGHT_INDEX) == nullptr,
    //         OP_LOGE(A_INNER_DEBUG, "GetOptionalInputShape of mm_weight is null."),
    //         return ge::GRAPH_FAILED);
    // }

    // OP_TILING_CHECK(CheckMmShapeDims(context) != ge::GRAPH_SUCCESS,
    //     OP_LOGE(A_INNER_DEBUG, "Check mm shape dim failed!"),
    //     return ge::GRAPH_FAILED);

    // if (tilingData->commonTilingInfo.isPermuteOut) {
    //     if (context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) == nullptr ||
    //         context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() == NUM_ZERO) {
    //         OP_LOGE(A_INNER_DEBUG, "The permuteOut should not be null when permuteOutFlag is true!");
    //         return ge::GRAPH_FAILED;
    //     }
    //     if (context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
    //         OP_LOGE(A_INNER_DEBUG,
    //             "The dim of permuteOut(A, H1) should be 2, but got %lu!",
    //             context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum());
    //         return ge::GRAPH_FAILED;
    //     }
    // } else {
    //     if (context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) != nullptr &&
    //         context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() != NUM_ZERO) {
    //         OP_LOGE(A_INNER_DEBUG, "The permuteOut should be null when permuteOutFlag is false!");
    //         return ge::GRAPH_FAILED;
    //     }
    // }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmChecker::CheckQuantDType(const gert::TilingContext *context) const // 只对quant数据的类型做校验
{
    // OP_TILING_CHECK(
    //     (context->GetInputDesc(GMM_X_INDEX) == nullptr) || (context->GetInputDesc(GMM_WEIGHT_INDEX) == nullptr),
    //     OP_LOGE(A_INNER_DEBUG, "GetInputDesc gmmX or gmmWeight returned null."),
    //     return ge::GRAPH_FAILED);
    // OP_TILING_CHECK(context->GetOutputDesc(OUTPUT_Y_INDEX) == nullptr,
    //     OP_LOGE(A_INNER_DEBUG, "GetOutputDesc y returned null."),
    //     return ge::GRAPH_FAILED);
    // OP_TILING_CHECK((context->GetInputDesc(GMMX_SCALE_INDEX)->GetDataType() != ge::DT_FLOAT),
    //     OP_LOGE(A_INNER_DEBUG, "Unsupported dataType, gmmx scale only support float32!"),
    //     return ge::GRAPH_FAILED);
    // OP_TILING_CHECK((context->GetInputDesc(GMMW_SCALE_INDEX)->GetDataType() != ge::DT_FLOAT),
    //     OP_LOGE(A_INNER_DEBUG, "Unsupported dataType, gmmweight scale only support float32!!"),
    //     return ge::GRAPH_FAILED);
    // OP_TILING_CHECK((context->GetInputDesc(GMM_X_INDEX)->GetDataType() != ge::DT_HIFLOAT8),
    //     OP_LOGE(A_INNER_DEBUG, "Unsupported dataType, quant gmmx only support hifloat8!"),
    //     return ge::GRAPH_FAILED);
    // OP_TILING_CHECK(
    //     (context->GetInputDesc(GMM_X_INDEX)->GetDataType() != context->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType()) ||
    //         (context->GetInputDesc(GMM_X_INDEX)->GetDataType() !=
    //             context->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType()),
    //     OP_LOGE(A_INNER_DEBUG, "The dataType of gmmWeight and gmmY should be the same with gmmX."),
    //     return ge::GRAPH_FAILED);
    // if (tilingData->commonTilingInfo.isNeedMM) {
    //     auto mmXDex = context->GetOptionalInputDesc(MM_X_INDEX);
    //     OP_TILING_CHECK(mmXDex == nullptr,
    //         OP_LOGE(A_INNER_DEBUG, "Flag isNeedMM is True, but MM_X is null."),
    //         return ge::GRAPH_FAILED);
    //     auto mmWeightDesc = context->GetOptionalInputDesc(MM_WEIGHT_INDEX);
    //     OP_TILING_CHECK(mmWeightDesc == nullptr,
    //         OP_LOGE(A_INNER_DEBUG, "Flag isNeedMM is True, MM_WEIGHT is null."),
    //         return ge::GRAPH_FAILED);
    //     auto mmYDesc = context->GetOutputDesc(OUTPUT_MM_Y_INDEX);
    //     OP_TILING_CHECK(
    //         mmYDesc == nullptr, OP_LOGE(A_INNER_DEBUG, "GetOutputDesc mmY returned null."), return ge::GRAPH_FAILED);
    //     if ((context->GetInputDesc(MMX_SCALE_INDEX)!=nullptr) &&
    //         (context->GetInputDesc(MMW_SCALE_INDEX)!=nullptr)){
    //         OP_TILING_CHECK((context->GetInputDesc(MMX_SCALE_INDEX)->GetDataType() != ge::DT_FLOAT),
    //             OP_LOGE(A_INNER_DEBUG, "Unsupported dataType, mmx scale only support float32!"),
    //             return ge::GRAPH_FAILED);
    //         OP_TILING_CHECK((context->GetInputDesc(MMW_SCALE_INDEX)->GetDataType() != ge::DT_FLOAT),
    //             OP_LOGE(A_INNER_DEBUG, "Unsupported dataType, mmweight scale only support float32!!"),
    //             return ge::GRAPH_FAILED);    
    //         }
    //     OP_TILING_CHECK((context->GetInputDesc(MM_X_INDEX)->GetDataType() != ge::DT_HIFLOAT8),
    //         OP_LOGE(A_INNER_DEBUG, "Unsupported dataType, quant mmx only support hifloat8!"),
    //         return ge::GRAPH_FAILED);
    //     OP_TILING_CHECK((context->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() !=
    //                         context->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType()) ||
    //                         (context->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() !=
    //                             context->GetOutputDesc(OUTPUT_MM_Y_INDEX)->GetDataType()),
    //         OP_LOGE(A_INNER_DEBUG, "The dataType of mmWeight and mmY should be the same with mmX."),
    //         return ge::GRAPH_FAILED);
    // }

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling