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
    } // namespace input
    struct output {
        enum : uint32_t {
            // REQUIRED
            OUTPUT_X_INDEX = 0,
            COUNT
        };
    } // namespace output
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
    }
}

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
    } // namespace input
    struct output {
    enum : uint32_t {
        OUTPUT_X_INDEX = 0,
        COUNT
    };
    } // namespace output
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
    }
}

constexpr uint32_t INT8_COMM_QUANT = 2U;
constexpr uint64_t INIT_TILINGKEY = 10000;
constexpr uint64_t TILINGKEY_TP_WORLD_SIZE = 100;
constexpr uint64_t TP_WORLD_SIZE_TWO = 2;
constexpr uint32_t TILINGKEY_INT8_COMM_QUANT = 20U;

constexpr uint32_t THREE_DIMS = 3U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;
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
static bool CheckInputTensorDim_1(const gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(Idx::input::EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE(nodeName, "expandX is null."), return false);
    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expandX must be 2-dimension, but got %lu dim",
        expandXStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(Idx::input::EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsStorageShape == nullptr, OP_LOGE(nodeName, "expertIds is null."), return false);
    OP_TILING_CHECK(expertIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expertIds must be 2-dimension, but got %lu dim",
        expertIdsStorageShape->GetStorageShape().GetDimNum()), return false);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "expertIds dim0 = %ld", expertIdsDim0);
    OP_LOGD(nodeName, "expertIds dim1 = %ld", expertIdsDim1);

    const gert::StorageShape *assistInfoStorageShape = context->GetInputShape(Idx::input::ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoStorageShape == nullptr, OP_LOGE(nodeName, "assistInfoForCombine is null."), return false);
    OP_TILING_CHECK(assistInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "assistInfoForCombine must be 1-dimension, but got %lu dim",
        assistInfoStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "assistInfoForCombine dim0 = %ld", assistInfoStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *epSendCountsStorageShape = context->GetInputShape(Idx::input::EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsStorageShape == nullptr, OP_LOGE(nodeName, "epSendCounts is null."), return false);
    OP_TILING_CHECK(epSendCountsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "epSendCounts must be 1-dimension, but got %lu dim",
        epSendCountsStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "epSendCounts dim0 = %ld", epSendCountsStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(Idx::input::EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesStorageShape == nullptr, OP_LOGE(nodeName, "expertScales is null."), return false);
    OP_TILING_CHECK(expertScalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expertScales must be 2-dimension, but got %lu dim",
        expertScalesStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expertScales dim0 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertScales dim1 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(1));
    return true;
}



} // namespace common_const

#endif