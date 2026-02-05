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


namespace optiling {
    namespace index{
        constexpr uint32_t OP_VERSION_2 = 2;
        namespace input{
            enum : uint32_t{
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
        }
        namespace output{
            enum : uint32_t{
                // REQUIRED
                OUTPUT_X_INDEX = 0,
                COUNT
            };
        }
        namespace attr{
            enum : uint32_t{
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

    namespace index_extend{
        constexpr uint32_t OP_VERSION_2 = 2;
        namespace input{
            enum : uint32_t{
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
        }
        namespace output{
            enum : uint32_t{
                OUTPUT_X_INDEX = 0,
                COUNT
            };
        }
        namespace attr{
            enum : uint32_t{
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
    constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U; // numeric representation of AlltoAll
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
    const char *K_INNER_DEBUG = "MoeDistributeCombineV2 Tiling Debug";

    enum class CommQuantMode : int32_t {
        NON_QUANT = 0,
        INT12_QUANT = 1,
        INT8_QUANT = 2
    };
    using CommQuantModeType = std::underlying_type_t<CommQuantMode>;
}