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
 * \file moe_distribute_combine_tiling_base.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_TILING_BASE_H
#define MOE_DISTRIBUTE_COMBINE_TILING_BASE_H

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include "moe_distribute_combine_tiling_base.h"
#include "tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../op_kernel/moe_distribute_combine_tiling.h"
#include "arch35/moe_distribute_combine_tiling_arch35.h"
#include "../../op_kernel/moe_distribute_combine_v2_tiling.h"
#include "../../op_kernel/moe_distribute_combine_v2_tiling_key.h"
#include "mc2_hcom_topo_info.h"

using namespace Mc2Tiling;

namespace common_const {
struct Index {
    static constexpr uint32_t OP_VERSION_2 = 2;
    struct input {
        enum : uint32_t {
            // REQUIRED
            EXPAND_X_INDEX,
            EXPERT_IDS_INDEX,
            ASSIST_INFO_INDEX,
            EP_SEND_COUNTS_INDEX,
            EXPERT_SCALES_INDEX,
            // OPTIONAL
            TP_SEND_COUNTS_INDEX,
            X_ACTIVE_MASK_INDEX,
            ACTIVATION_SCALE_INDEX,
            WEIGHT_SCALE_INDEX,
            GROUP_LIST_INDEX,
            SHARED_EXPERT_X_INDEX,
            ELASTIC_INFO_INDEX,
            ORI_X_INDEX,
            CONST_EXPERT_ALPHA_1_INDEX,
            CONST_EXPERT_ALPHA_2_INDEX,
            CONST_EXPERT_V_INDEX,
            PERFORMANCE_INFO_INDEX,
            COUNT
        };
    }; // namespace input
    struct output {
        enum : uint32_t {
            // REQUIRED
            OUTPUT_X_INDEX = 0,
            COUNT
        };
    }; // namespace output
    struct attr {
        enum : uint32_t {
            // REQUIRED
            ATTR_GROUP_EP_INDEX,
            ATTR_EP_WORLD_SIZE_INDEX,
            ATTR_EP_RANK_ID_INDEX,
            ATTR_MOE_EXPERT_NUM_INDEX,
            // OPTIONAL
            ATTR_GROUP_TP_INDEX,
            ATTR_TP_WORLD_SIZE_INDEX,
            ATTR_TP_RANK_ID_INDEX,
            ATTR_EXPERT_SHARD_TYPE_INDEX,
            ATTR_SHARED_EXPERT_NUM_INDEX,
            ATTR_SHARED_EXPERT_RANK_NUM_INDEX,
            ATTR_GLOBAL_BS_INDEX,
            ATTR_OUT_DTYPE_INDEX,
            ATTR_COMM_QUANT_MODE_INDEX,
            ATTR_GROUP_LIST_TYPE_INDEX,
            ATTR_COMM_ALG_INDEX,
            ATTR_ZERO_EXPERT_NUM_INDEX,
            ATTR_COPY_EXPERT_NUM_INDEX,
            ATTR_CONST_EXPERT_NUM_INDEX,
            COUNT
        };
    };
};

struct IndexExtend
{
    static constexpr uint32_t OP_VERSION_2 = 2;
    struct input {
    enum : uint32_t {
        // REQUIRED
        EXPAND_X_INDEX,
        EXPERT_IDS_INDEX,
        ASSIST_INFO_INDEX,
        EP_SEND_COUNTS_INDEX,
        EXPERT_SCALES_INDEX,
        MC2_CONTEXT,
        // OPTIONAL
        TP_SEND_COUNTS_INDEX,
        X_ACTIVE_MASK_INDEX,
        ACTIVATION_SCALE_INDEX,
        WEIGHT_SCALE_INDEX,
        GROUP_LIST_INDEX,
        SHARED_EXPERT_X_INDEX,
        ELASTIC_INFO_INDEX,
        ORI_X_INDEX,
        CONST_EXPERT_ALPHA_1_INDEX,
        CONST_EXPERT_ALPHA_2_INDEX,
        CONST_EXPERT_V_INDEX,
        PERFORMANCE_INFO_INDEX,
        COUNT
    };
    }; // namespace input
    struct output {
    enum : uint32_t {
        OUTPUT_X_INDEX = 0,
        COUNT
    };
    }; // namespace output
    struct attr {
    enum : uint32_t {
        // REQUIRED
        ATTR_GROUP_EP_INDEX,
        ATTR_EP_WORLD_SIZE_INDEX,
        ATTR_EP_RANK_ID_INDEX,
        ATTR_MOE_EXPERT_NUM_INDEX,
        HCCL_BUFF_SIZE,
        HCCL_TOPO_TYPE,
        // OPTIONAL
        ATTR_GROUP_TP_INDEX,
        ATTR_TP_WORLD_SIZE_INDEX,
        ATTR_TP_RANK_ID_INDEX,
        ATTR_EXPERT_SHARD_TYPE_INDEX,
        ATTR_SHARED_EXPERT_NUM_INDEX,
        ATTR_SHARED_EXPERT_RANK_NUM_INDEX,
        ATTR_GLOBAL_BS_INDEX,
        ATTR_OUT_DTYPE_INDEX,
        ATTR_COMM_QUANT_MODE_INDEX,
        ATTR_GROUP_LIST_TYPE_INDEX,
        ATTR_COMM_ALG_INDEX,
        ATTR_ZERO_EXPERT_NUM_INDEX,
        ATTR_COPY_EXPERT_NUM_INDEX,
        ATTR_CONST_EXPERT_NUM_INDEX,
        COUNT
    };
    };
};

constexpr uint32_t INT8_COMM_QUANT = 2U;
constexpr uint64_t INIT_TILINGKEY = 10000;
constexpr uint64_t TILINGKEY_TP_WORLD_SIZE = 100;
constexpr uint64_t TP_WORLD_SIZE_TWO = 2;
constexpr uint32_t TILINGKEY_INT8_COMM_QUANT = 20U;

constexpr uint32_t THREE_DIMS = 3U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIMS = 1U;
constexpr uint32_t ASSIST_INFO_DIMS = 1U;
constexpr uint64_t TILING_KEY_BASE_A2 = 2000UL;
constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 3000UL;
constexpr uint64_t TILING_KEY_INT8_COMM_QUANT_A2 = 100UL;
constexpr uint32_t ARR_LENGTH = 128U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;     // numeric representation of AlltoAll
constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7U; // numeric representation of AlltoAll
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t ALIGNED_LEN = 256U;
constexpr uint32_t DTYPE_SIZE_HALF = 2;
constexpr uint8_t BUFFER_SINGLE = 1;
constexpr uint8_t BUFFER_NUM = 2;

constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MAX_EP_WORLD_SIZE = 768L; // 384 * 2
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t EP_RESTRICT_8 = 8;
constexpr int64_t MAX_TP_WORLD_SIZE = 2;
constexpr int64_t BS_UPPER_BOUND = 512;

constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr size_t MASK_CALC_NEED_WORKSPACE = 10UL * 1024UL;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
constexpr uint32_t VERSION_2 = 2;
constexpr uint32_t HCOMMCNT_2 = 2;
constexpr uint32_t RANK_LIST_NUM = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
constexpr int64_t LOCAL_EXPERT_MAX_SIZE = 2048;
constexpr int64_t K_MAX = 16;
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 8192;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t TRIPLE = 3;
constexpr uint64_t ASSIST_NUM_PER_A = 128UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t SCALE_EXPAND_IDX_BUFFER = 44UL; // scale32B + 3*4expandIdx
constexpr uint64_t DOUBLE_DATA_BUFFER = 2UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr int64_t ELASTIC_METAINFO_OFFSET = 4;
// A2
constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 384;
constexpr int32_t MAX_EP_WORLD_SIZE_A2_LAYERED = 64;
constexpr int32_t MAX_MOE_EXPERT_NUMS_A2 = 512;
constexpr uint32_t MAX_HIDDEN_SIZE_A2 = 7168;
constexpr uint32_t LAYERED_MAX_HIDDEN_SIZE_A2 = 10240;
constexpr uint32_t MAX_BATCH_SIZE_A2 = 256;
constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
constexpr uint32_t BLOCK_SIZE_A2 = 32;
constexpr uint32_t MAX_K_VALUE_A2 = 16;
}


namespace optiling {

template <class Idx>
static bool CheckInputTensorDim(const gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(Idx::input::EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE(nodeName, "expandX is null."), return false);
    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != common_const::TWO_DIMS,
        OP_LOGE(nodeName, "expandX must be 2-dimension, but got %lu dim",
        expandXStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(Idx::input::EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsStorageShape == nullptr, OP_LOGE(nodeName, "expertIds is null."), return false);
    OP_TILING_CHECK(expertIdsStorageShape->GetStorageShape().GetDimNum() != common_const::TWO_DIMS,
        OP_LOGE(nodeName, "expertIds must be 2-dimension, but got %lu dim",
        expertIdsStorageShape->GetStorageShape().GetDimNum()), return false);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "expertIds dim0 = %ld", expertIdsDim0);
    OP_LOGD(nodeName, "expertIds dim1 = %ld", expertIdsDim1);

    const gert::StorageShape *assistInfoStorageShape = context->GetInputShape(Idx::input::ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoStorageShape == nullptr, OP_LOGE(nodeName, "assistInfoForCombine is null."), return false);
    OP_TILING_CHECK(assistInfoStorageShape->GetStorageShape().GetDimNum() != common_const::ONE_DIMS,
        OP_LOGE(nodeName, "assistInfoForCombine must be 1-dimension, but got %lu dim",
        assistInfoStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "assistInfoForCombine dim0 = %ld", assistInfoStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *epSendCountsStorageShape = context->GetInputShape(Idx::input::EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsStorageShape == nullptr, OP_LOGE(nodeName, "epSendCounts is null."), return false);
    OP_TILING_CHECK(epSendCountsStorageShape->GetStorageShape().GetDimNum() != common_const::ONE_DIMS,
        OP_LOGE(nodeName, "epSendCounts must be 1-dimension, but got %lu dim",
        epSendCountsStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "epSendCounts dim0 = %ld", epSendCountsStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(Idx::input::EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesStorageShape == nullptr, OP_LOGE(nodeName, "expertScales is null."), return false);
    OP_TILING_CHECK(expertScalesStorageShape->GetStorageShape().GetDimNum() != common_const::TWO_DIMS,
        OP_LOGE(nodeName, "expertScales must be 2-dimension, but got %lu dim",
        expertScalesStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expertScales dim0 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertScales dim1 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(1));

    return true;
}

template <class Idx>
static bool CheckOptionalInputTensorDim(const gert::TilingContext *context, const char *nodeName,
    const bool isActiveMask, const bool hasElasticInfo, const bool isPerformance, uint32_t tpWorldSize)
{
    const gert::StorageShape* oriXStorageShape = context->GetOptionalInputShape(Idx::input::ORI_X_INDEX);
    if (oriXStorageShape != nullptr) {
        OP_TILING_CHECK(
            oriXStorageShape->GetStorageShape().GetDimNum() != common_const::TWO_DIMS,
            OP_LOGE(
                nodeName, "ori_x must be 2-dimension, but got %lu dim",
                oriXStorageShape->GetStorageShape().GetDimNum()),
            return false);
    }

    const gert::StorageShape* constExpertAlpha1StorageShape =
        context->GetOptionalInputShape(Idx::input::CONST_EXPERT_ALPHA_1_INDEX);
    if (constExpertAlpha1StorageShape != nullptr) {
        OP_TILING_CHECK(
            constExpertAlpha1StorageShape->GetStorageShape().GetDimNum() != common_const::TWO_DIMS,
            OP_LOGE(
                nodeName, "const_expert_alpha_1 must be 2-dimension, but got %lu dim",
                constExpertAlpha1StorageShape->GetStorageShape().GetDimNum()),
            return false);
    }

    const gert::StorageShape* constExpertAlpha2StorageShape =
        context->GetOptionalInputShape(Idx::input::CONST_EXPERT_ALPHA_2_INDEX);
    if (constExpertAlpha2StorageShape != nullptr) {
        OP_TILING_CHECK(
            constExpertAlpha2StorageShape->GetStorageShape().GetDimNum() != common_const::TWO_DIMS,
            OP_LOGE(
                nodeName, "const_expert_alpha_2 must be 2-dimension, but got %lu dim",
                constExpertAlpha2StorageShape->GetStorageShape().GetDimNum()),
            return false);
    }

    const gert::StorageShape* constExpertVStorageShape = context->GetOptionalInputShape(Idx::input::CONST_EXPERT_V_INDEX);
    if (constExpertVStorageShape != nullptr) {
        OP_TILING_CHECK(
            constExpertVStorageShape->GetStorageShape().GetDimNum() != common_const::TWO_DIMS,
            OP_LOGE(
                nodeName, "const_expert_v must be 2-dimension, but got %lu dim",
                constExpertVStorageShape->GetStorageShape().GetDimNum()),
            return false);
    }

    if (tpWorldSize == common_const::TP_WORLD_SIZE_TWO) {
        const gert::StorageShape *tpSendCountsStorageShape = context->GetOptionalInputShape(Idx::input::TP_SEND_COUNTS_INDEX);
        OP_TILING_CHECK(tpSendCountsStorageShape == nullptr, OP_LOGE(nodeName, "tpSendCounts is null."), return false);
        OP_TILING_CHECK(tpSendCountsStorageShape->GetStorageShape().GetDimNum() != common_const::ONE_DIMS,
            OP_LOGE(nodeName, "tpSendCounts must be 1-dimension, but got %lu dim",
            tpSendCountsStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "tpSendCounts dim0 = %ld", tpSendCountsStorageShape->GetStorageShape().GetDim(0));
    }

    if (isActiveMask) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(Idx::input::X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskStorageShape == nullptr, OP_LOGE(nodeName, "xActiveMask is null."), return false);
        const int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(((xActiveMaskDimNums != common_const::ONE_DIMS) && (xActiveMaskDimNums != common_const::TWO_DIMS)),
            OP_LOGE(nodeName, "xActiveMask must be 1-dimension or 2-dimension, but got %ld dim",
            xActiveMaskDimNums), return false);
    }
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(Idx::input::ELASTIC_INFO_INDEX);
        OP_TILING_CHECK(elasticInfoStorageShape == nullptr, OP_LOGE(nodeName, "elasticInfo is null."), return false);
        OP_TILING_CHECK(elasticInfoStorageShape->GetStorageShape().GetDimNum() != common_const::ONE_DIMS,
            OP_LOGE(nodeName, "elasticInfo dim must be 1, but current dim num is %lu.",
            elasticInfoStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "elasticInfo dim0 = %ld", elasticInfoStorageShape->GetStorageShape().GetDim(0));
    }

    if (isPerformance) {
        const gert::StorageShape *performanceInfoStorageShape = context->GetOptionalInputShape(Idx::input::PERFORMANCE_INFO_INDEX);
        OP_TILING_CHECK(performanceInfoStorageShape == nullptr, OP_LOGE(nodeName, "performanceInfo is null."), return false);
        OP_TILING_CHECK(performanceInfoStorageShape->GetStorageShape().GetDimNum() != common_const::ONE_DIMS,
            OP_LOGE(nodeName, "performanceInfo dim must be 1, but current dim num is %lu.",
            performanceInfoStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "performanceInfo dim0 = %ld", performanceInfoStorageShape->GetStorageShape().GetDim(0));
 	}

    const gert::StorageShape *activationScaleStorageShape = context->GetOptionalInputShape(Idx::input::ACTIVATION_SCALE_INDEX);
    OP_TILING_CHECK(activationScaleStorageShape != nullptr, OP_LOGE(nodeName, "activationScale is not null."), return false);

    const gert::StorageShape *weightScaleStorageShape = context->GetOptionalInputShape(Idx::input::WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK(weightScaleStorageShape != nullptr, OP_LOGE(nodeName, "weightScale is not null."), return false);

    const gert::StorageShape *groupListStorageShape = context->GetOptionalInputShape(Idx::input::GROUP_LIST_INDEX);
    OP_TILING_CHECK(groupListStorageShape != nullptr, OP_LOGE(nodeName, "groupList is not null."), return false);

    const gert::StorageShape *sharedExpertX = context->GetOptionalInputShape(Idx::input::SHARED_EXPERT_X_INDEX);
    if (sharedExpertX != nullptr) {
        auto attrs = context->GetAttrs();
        auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(Idx::attr::ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
        OP_TILING_CHECK(*sharedExpertRankNumPtr != 0, OP_LOGE(nodeName, "sharedExpertX only support input None "\
            "when sharedExpertRankNum is non-zero."), return false);
        OP_TILING_CHECK(((sharedExpertX->GetStorageShape().GetDimNum() != common_const::TWO_DIMS) &&
                        (sharedExpertX->GetStorageShape().GetDimNum() != common_const::THREE_DIMS)),
                        OP_LOGE(nodeName, "sharedExpertX must be 2-dimension or 3-dimension, but got %lu dim",
                                sharedExpertX->GetStorageShape().GetDimNum()), return false);
    }

    return true;
}

template <class Idx>
static bool CheckOutputTensorDim(const gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetOutputShape(Idx::output::OUTPUT_X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "x is null."), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != common_const::TWO_DIMS,
        OP_LOGE(nodeName, "x must be 2-dimension, but got %lu dim", xStorageShape->GetStorageShape().GetDimNum()),
        return false);
    OP_LOGD(nodeName, "x dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "x dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));

    return true;
}

template <class Idx>
static bool CheckTensorDim(gert::TilingContext *context, const char *nodeName, const bool isActiveMask,
                           const bool hasElasticInfo, const bool isPerformance, uint32_t tpWorldSize)
{
    OP_TILING_CHECK(!CheckInputTensorDim<Idx>(context, nodeName),
        OP_LOGE(nodeName, "param shape of input tensor is invalid"), return false);

    OP_TILING_CHECK(!CheckOptionalInputTensorDim<Idx>(context, nodeName, isActiveMask, hasElasticInfo, isPerformance, tpWorldSize),
        OP_LOGE(nodeName, "param shape of optional input tensor is invalid"), return false);

    OP_TILING_CHECK(!CheckOutputTensorDim<Idx>(context, nodeName),
        OP_LOGE(nodeName, "param shape of output tensor is invalid"), return false);

    return true;
}

// 校验数据类型
template <class Idx>
static bool CheckTensorDataType(const gert::TilingContext *context, const char *nodeName, const bool isActiveMask,
                                const bool hasElasticInfo, const bool isPerformance, uint32_t tpWorldSize)
{
    auto expandXDesc = context->GetInputDesc(Idx::input::EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return false);
    OP_TILING_CHECK((expandXDesc->GetDataType() != ge::DT_BF16) && (expandXDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be bf16 or float16, but is %s",
        Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return false);

    auto oriXDesc = context->GetOptionalInputDesc(Idx::input::ORI_X_INDEX);
    if (oriXDesc != nullptr) {
        OP_TILING_CHECK(
            (oriXDesc->GetDataType() != expandXDesc->GetDataType()),
            OP_LOGE(
                nodeName, "ori_x dataType is invalid, dataType should be same as expandX dataType as %s, but now is %s",
                Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                Ops::Base::ToString(oriXDesc->GetDataType()).c_str()),
            return false);
    }

    auto constExpertAlpha1Desc = context->GetOptionalInputDesc(Idx::input::CONST_EXPERT_ALPHA_1_INDEX);
    if (constExpertAlpha1Desc != nullptr) {
        OP_TILING_CHECK(
            (constExpertAlpha1Desc->GetDataType() != expandXDesc->GetDataType()),
            OP_LOGE(
                nodeName, "const_expert_alpha_1 dataType is invalid, dataType should be same as expandX dataType as %s, but now is %s",
                Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                Ops::Base::ToString(constExpertAlpha1Desc->GetDataType()).c_str()),
            return false);
    }

    auto constExpertAlpha2Desc = context->GetOptionalInputDesc(Idx::input::CONST_EXPERT_ALPHA_2_INDEX);
    if (constExpertAlpha2Desc != nullptr) {
        OP_TILING_CHECK(
            (constExpertAlpha2Desc->GetDataType() != expandXDesc->GetDataType()),
            OP_LOGE(
                nodeName, "const_expert_alpha_2 dataType is invalid, dataType should be same as expandX dataType as %s, but now is %s",
                Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                Ops::Base::ToString(constExpertAlpha2Desc->GetDataType()).c_str()),
            return false);
    }

    auto constExpertVDesc = context->GetOptionalInputDesc(Idx::input::CONST_EXPERT_V_INDEX);
    if (constExpertVDesc != nullptr) {
        OP_TILING_CHECK(
            (constExpertVDesc->GetDataType() != expandXDesc->GetDataType()),
            OP_LOGE(
                nodeName, "const_expert_v dataType is invalid, dataType should be same as expandX dataType as %s, but now is %s",
                Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                Ops::Base::ToString(constExpertVDesc->GetDataType()).c_str()),
            return false);
    }

    auto expertIdsDesc = context->GetInputDesc(Idx::input::EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE(nodeName, "expertIdsDesc is null."), return false);
    OP_TILING_CHECK((expertIdsDesc->GetDataType() != ge::DT_INT32), OP_LOGE(nodeName, "expertIds dataType is invalid, "
        "dataType should be int32, but is %s", Ops::Base::ToString(expertIdsDesc->GetDataType()).c_str()), return false);
    auto assistInfoDesc = context->GetInputDesc(Idx::input::ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK((assistInfoDesc->GetDataType() != ge::DT_INT32), OP_LOGE(nodeName, "assistInfoForCombine dataType is invalid,"
        " dataType should be int32, but is %s", Ops::Base::ToString(assistInfoDesc->GetDataType()).c_str()), return false);
    auto epSendCountsDesc = context->GetInputDesc(Idx::input::EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsDesc == nullptr, OP_LOGE(nodeName, "epSendCountsDesc is null."), return false);
    OP_TILING_CHECK((epSendCountsDesc->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName, "epSendCounts dataType is invalid, dataType should be int32, but is %s",
        Ops::Base::ToString(epSendCountsDesc->GetDataType()).c_str()), return false);
    if (tpWorldSize == common_const::TP_WORLD_SIZE_TWO) {
        auto tpSendCountsDesc = context->GetOptionalInputDesc(Idx::input::TP_SEND_COUNTS_INDEX);
        OP_TILING_CHECK(tpSendCountsDesc == nullptr, OP_LOGE(nodeName, "tpSendCountsDesc is null."), return false);
        OP_TILING_CHECK((tpSendCountsDesc->GetDataType() != ge::DT_INT32),
            OP_LOGE(nodeName, "tpSendCounts dataType is invalid, dataType should be int32, but is %s",
            Ops::Base::ToString(tpSendCountsDesc->GetDataType()).c_str()), return false);
    }
    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(Idx::input::X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
        OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL, OP_LOGE(nodeName, "xActiveMask dataType is invalid,"
            " dataType should be bool, but is %s.", Ops::Base::ToString(xActiveMaskDesc->GetDataType()).c_str()), return false);
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(Idx::input::ELASTIC_INFO_INDEX);
        OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE(nodeName, "elasticInfoDesc is null."), return false);
        OP_TILING_CHECK(elasticInfoDesc->GetDataType() != ge::DT_INT32, OP_LOGE(nodeName,
            "elasticInfoDesc dataType is invalid, dataType should be int32, but is %s.",
            Ops::Base::ToString(elasticInfoDesc->GetDataType()).c_str()), return false);
    }
    if (isPerformance) {
        auto performanceInfoDesc = context->GetOptionalInputDesc(Idx::input::PERFORMANCE_INFO_INDEX);
        OP_TILING_CHECK(performanceInfoDesc->GetDataType() != ge::DT_INT64, OP_LOGE(nodeName,
            "performanceInfoDesc dataType is invalid, dataType should be int64, but is %s.",
            Ops::Base::ToString(performanceInfoDesc->GetDataType()).c_str()), return false);
    }
    auto sharedExpertXDesc = context->GetOptionalInputDesc(Idx::input::SHARED_EXPERT_X_INDEX);
    if (sharedExpertXDesc != nullptr) {
        OP_TILING_CHECK(sharedExpertXDesc->GetDataType() != expandXDesc->GetDataType(),
            OP_LOGE(nodeName, "sharedExpertX dataType should be the same as expandX dataType, but got sharedExpertX"
            "dataType %s, expandX dataType %s.", Ops::Base::ToString(sharedExpertXDesc->GetDataType()).c_str(),
            Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return false);
    }
    auto expertScalesDesc = context->GetInputDesc(Idx::input::EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE(nodeName, "expertScalesDesc is null."), return false);
    OP_TILING_CHECK((expertScalesDesc->GetDataType() != ge::DT_FLOAT),
        OP_LOGE(nodeName, "expertScales dataType is invalid, dataType should be float, but is %s",
        Ops::Base::ToString(expertScalesDesc->GetDataType()).c_str()), return false);
    auto xDesc = context->GetOutputDesc(Idx::output::OUTPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK((xDesc->GetDataType() != expandXDesc->GetDataType()), OP_LOGE(nodeName,
        "x dataType is invalid, dataType should be equal to expandX dataType %s, but is %s",
        Ops::Base::ToString(expandXDesc->GetDataType()).c_str(), Ops::Base::ToString(xDesc->GetDataType()).c_str()),
        return false);
    return true;
}

template <class Idx>
static bool CheckTensorFormat(const gert::TilingContext *context, const char *nodeName, const bool isActiveMask,
                              const bool hasElasticInfo, const bool isPerformance, uint32_t tpWorldSize)
{
    auto oriXDesc = context->GetOptionalInputDesc(Idx::input::ORI_X_INDEX);
    if (oriXDesc != nullptr) {
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(oriXDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE(nodeName, "ori_x Format is invalid"), return false);
    }

    auto constExpertAlpha1Desc = context->GetOptionalInputDesc(Idx::input::CONST_EXPERT_ALPHA_1_INDEX);
    if (constExpertAlpha1Desc != nullptr) {
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(constExpertAlpha1Desc->GetStorageFormat())) ==
                ge::FORMAT_FRACTAL_NZ,
            OP_LOGE(nodeName, "const_expert_alpha_1 Format is invalid"), return false);
    }

    auto constExpertAlpha2Desc = context->GetOptionalInputDesc(Idx::input::CONST_EXPERT_ALPHA_2_INDEX);
    if (constExpertAlpha2Desc != nullptr) {
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(constExpertAlpha2Desc->GetStorageFormat())) ==
                ge::FORMAT_FRACTAL_NZ,
            OP_LOGE(nodeName, "const_expert_alpha_2 Format is invalid"), return false);
    }

    auto constExpertVDesc = context->GetOptionalInputDesc(Idx::input::CONST_EXPERT_V_INDEX);
    if (constExpertVDesc != nullptr) {
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(constExpertVDesc->GetStorageFormat())) ==
                ge::FORMAT_FRACTAL_NZ,
            OP_LOGE(nodeName, "const_expert_v Format is invalid"), return false);
    }

    auto expandXDesc = context->GetInputDesc(Idx::input::EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expandXFormat is invalid"), return false);

    auto expertIdsDesc = context->GetInputDesc(Idx::input::EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE(nodeName, "expertIdsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expertIdsFormat is invalid"), return false);

    auto assistInfoDesc = context->GetInputDesc(Idx::input::ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(assistInfoDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "assistInfoFormat is invalid"), return false);

    auto epSendCountsDesc = context->GetInputDesc(Idx::input::EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsDesc == nullptr, OP_LOGE(nodeName, "epSendCountsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(epSendCountsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "epSendCountsFormat is invalid"), return false);

    if (tpWorldSize == common_const::TP_WORLD_SIZE_TWO) {
        auto tpSendCountsDesc = context->GetOptionalInputDesc(Idx::input::TP_SEND_COUNTS_INDEX);
        OP_TILING_CHECK(tpSendCountsDesc == nullptr, OP_LOGE(nodeName, "tpSendCountsDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(tpSendCountsDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "tpSendCountsFormat is invalid"), return false);
    }

    auto expertScalesDesc = context->GetInputDesc(Idx::input::EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE(nodeName, "expertScalesDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertScalesDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expertScalesFormat is invalid"), return false);

    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(Idx::input::X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "xActiveMaskFormat is invalid."), return false);
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(Idx::input::ELASTIC_INFO_INDEX);
        OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE(nodeName, "elasticInfoDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(elasticInfoDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "elasticInfo format is invalid."), return false);
    }
    if (isPerformance) {
 	         auto performanceInfoDesc = context->GetOptionalInputDesc(Idx::input::PERFORMANCE_INFO_INDEX);
 	         OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(performanceInfoDesc->GetStorageFormat())) ==
 	             ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "performanceInfoDesc format is invalid."), return false);
 	     }
    auto sharedExpertXDesc = context->GetOptionalInputDesc(Idx::input::SHARED_EXPERT_X_INDEX);
    OP_TILING_CHECK((sharedExpertXDesc != nullptr) &&
        (static_cast<ge::Format>(ge::GetPrimaryFormat(sharedExpertXDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ), OP_LOGE(nodeName, "sharedExpertXFormat is invalid."), return false);

    auto xDesc = context->GetOutputDesc(Idx::output::OUTPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "xFormat is invalid"), return false);

    return true;
}

template <class Idx>
static ge::graphStatus TilingCheckMoeDistributeCombine(gert::TilingContext *context, const char *nodeName,
    const bool isActiveMask, const bool hasElasticInfo, const bool isPerformance, uint32_t tpWorldSize)
{
    // 检查参数shape信息
    OP_TILING_CHECK(!CheckTensorDim<Idx>(context, nodeName, isActiveMask, hasElasticInfo, isPerformance, tpWorldSize),
                    OP_LOGE(nodeName, "param shape is invalid"), return ge::GRAPH_FAILED);
    // 检查参数dataType信息
    OP_TILING_CHECK(!CheckTensorDataType<Idx>(context, nodeName, isActiveMask, hasElasticInfo, isPerformance, tpWorldSize),
                    OP_LOGE(nodeName, "param dataType is invalid"), return ge::GRAPH_FAILED);
    // 检查参数format信息
    OP_TILING_CHECK(!CheckTensorFormat<Idx>(context, nodeName, isActiveMask, hasElasticInfo, isPerformance, tpWorldSize),
                    OP_LOGE(nodeName, "param Format is invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext *context, const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "get workspace failed"),
        return ge::GRAPH_FAILED);
    workspace[0] = common_const::SYSTEM_NEED_WORKSPACE + aivNum * common_const::MASK_CALC_NEED_WORKSPACE + 4*1024*1024*1024;
    OP_LOGD(nodeName, "workspace[0] size is %ld", workspace[0]);
    return ge::GRAPH_SUCCESS;
}

static uint64_t CalTilingKey(const uint32_t tpWorldSize, uint32_t commQuantMode)
{
    bool tp = false;
    uint32_t quantMode = TILINGKEY_NO_QUANT;
    uint32_t layeredMode = TILINGKEY_TPL_MTE;  // A2
    if (tpWorldSize == common_const::MAX_TP_WORLD_SIZE) {
        tp = true;
    }
    if (commQuantMode == common_const::INT8_COMM_QUANT) {
        quantMode = TILINGKEY_INT8_QUANT;
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(tp, quantMode, layeredMode, TILINGKEY_TPL_A3);
    return tilingKey;
}


} // namespace common_const

#endif