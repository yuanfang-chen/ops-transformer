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
 * \file allto_all_matmul_tiling_910.cpp
 * \brief
 */
#include "vector"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "op_mc2.h"
#include "mc2_hcom_topo_info.h"
#include "tiling/mc2_tiling_utils.h"
#include <map>
#include "allto_all_matmul_tiling_910.h"

using namespace AscendC;
using namespace ge;

namespace {
    const char *K_INNER_DEBUG = "AllToAllMatmul Tiling Debug";
    constexpr uint32_t BASE_K = 128;
    constexpr uint32_t ATTR_GROUP_INDEX = 0;
    constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1;
    constexpr uint64_t INIT_TILINGKEY = 1000000;
    constexpr uint64_t TILINGKEY_BIAS = 100000;
    constexpr uint32_t INPUT_X1_INDEX = 0;
    constexpr uint32_t INPUT_X2_INDEX = 1;
    constexpr uint32_t INPUT_BIAS_INDEX = 2;
    constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
    constexpr uint32_t USER_WORKSPACE_A2 = 1 * 1024 * 1024; // moeExpertNum_ * sizeof(uint32_t) + epWorldSize_ * 2 * 32
    constexpr uint32_t ELEMENT_SIZE = 2;
    constexpr uint32_t MAX_BLOCK_COUNT = 2;
    constexpr int32_t MIN_P_VALUE = 1;
    constexpr int32_t MAX_BUFF_BYTES = 204 * 1024 * 1024;
    constexpr int32_t FLAG_BUFF_BYTES = 5 * 512 * 1024;
    constexpr int32_t HALF_KBYTE = 512;
    constexpr int32_t UB_PINGPONG_SIZE = 2;
    constexpr int32_t MAX_UB_NUM = 97280;
    constexpr int32_t DEFAULT_COL = 256;
    constexpr int32_t DEFAULT_ROW = 128;
    constexpr int32_t SWIZZLE_DIRECT_ONE = 1;
    constexpr int32_t DEFAULT_SWIZZLE_COUNT = 7;
    constexpr int32_t SWIZZLE_COUNT_THREE = 3;
    constexpr int32_t CORE_NUM_FOUR = 4;
    constexpr int32_t CORE_NUM_EIGHT = 8;

    constexpr int32_t ALLTOALLMATMUL_TWO_RANK_FP16_FIRSTSTEPCORENUM_DEFAULT = 16;
    constexpr int32_t ALLTOALLMATMUL_TWO_RANK_FP16_PVALUE_DEFAULT = 14;
    constexpr int32_t ALLTOALLMATMUL_TWO_RANK_FP16_M0_DEFAULT = 128;
    constexpr int32_t ALLTOALLMATMUL_TWO_RANK_FP16_UBSIZE_DEFAULT = 3;

    constexpr int32_t ALLTOALLMATMUL_FOUR_RANK_FP16_FIRSTSTEPCORENUM_DEFAULT = 8;
    constexpr int32_t ALLTOALLMATMUL_FOUR_RANK_FP16_PVALUE_DEFAULT = 12;
    constexpr int32_t ALLTOALLMATMUL_FOUR_RANK_FP16_M0_DEFAULT = 128;
    constexpr int32_t ALLTOALLMATMUL_FOUR_RANK_FP16_UBSIZE_DEFAULT = 2;

    constexpr int32_t ALLTOALLMATMUL_EIGHT_RANK_FP16_PVALUE_DEFAULT = 12;
    constexpr int32_t ALLTOALLMATMUL_EIGHT_RANK_FP16_M0_DEFAULT = 128;
    constexpr int32_t ALLTOALLMATMUL_EIGHT_RANK_FP16_UBSIZE_DEFAULT = 2;

    constexpr int32_t WORLDSIZE_TWO = 2;
    constexpr int32_t WORLDSIZE_FOUR = 4;
    constexpr int32_t WORLDSIZE_EIGHT = 8;
    constexpr int32_t CONDITION_M_ST = 0;
    constexpr int32_t CONDITION_M_END = 1;
    constexpr int32_t CONDITION_K_ST = 2;
    constexpr int32_t CONDITION_K_END = 3;
    constexpr int32_t CONDITION_N_ST = 4;
    constexpr int32_t CONDITION_N_END = 5;
    constexpr uint32_t COUNT_PARAMS_WITH_BIAS = 4; // [x1, x2, bias, y]
    constexpr uint32_t COUNT_PARAMS_WITHOUT_BIAS = 3; // [x1, x2, y]
    const std::set<int> SUPPORT_RANK_SIZE_910{2, 4, 8};
    const std::vector<std::vector<uint32_t>> SUPPORTED_TYPES_WITH_BIAS = {
        {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16},
        {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}
    };
    const std::vector<std::vector<uint32_t>> SUPPORTED_TYPES_WITHOUT_BIAS = {
        {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16},
        {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}
    };
}

namespace MC2Tiling {

int32_t RoundNum(int32_t num, int32_t rnd) {
    if (rnd == 0) {
        return 0;
    }
    return (num + rnd - 1) / rnd * rnd;
}

template <typename T>
T ClampValue(T value, T minVal, T maxVal) {
    return (value < minVal) ? minVal : (value > maxVal) ? maxVal : value;
}

static std::map<int, std::vector<std::vector<int>>> g_alltoallmatmulTwoRankFP16UbsizeMap = {
    {2.0,
        {{-1, 7168, -1, 2816, -1, 1536}, {7168, 19456, 1664, 2816, -1, 1536},
        {-1, 19456, -1, 1152, 1536, 2816}, {19456, 2147483647, -1, 1152, 1536, 3072}}},
    {3.0,
        {{7168, 19456, -1, 1664, -1, 1536}}},
    {1.0,
        {{-1, 19456, 2816, 2147483647, -1, 1536}, {19456, 2147483647, -1, 2147483647, -1, 1536},
        {-1, 19456, -1, 1152, 2816, 2147483647}, {19456, 2147483647, -1, 1152, 3072, 2147483647},
        {-1, 2147483647, 1152, 2147483647, 1536, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_alltoallmatmulTwoRankFP16M0Map = {
    {128,
        {{-1, 5120, -1, 2147483647, -1, 5632}, {5120, 2147483647, 2816, 4608, -1, 1536},
        {5120, 2147483647, -1, 4608, 1536, 5632}, {5120, 11264, 4608, 2147483647, -1, 4608},
        {11264, 2147483647, 4608, 2147483647, -1, 5632}, {-1, 2147483647, -1, 2304, 5632, 2147483647},
        {-1, 2147483647, 2304, 4608, 5632, 6656}, {9216, 2147483647, 5632, 7680, 5632, 2147483647}}},
    {256,
        {{5120, 2147483647, -1, 2816, -1, 1536}, {5120, 11264, 4608, 2147483647, 4608, 5632},
        {-1, 2147483647, 4608, 5632, 5632, 6656}, {-1, 2147483647, 2304, 5632, 6656, 2147483647},
        {-1, 9216, 5632, 7680, 5632, 2147483647}, {-1, 2147483647, 7680, 2147483647, 5632, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_alltoallmatmulTwoRankFP16PvalueMap = {
    {14,
        {{-1, 7168, -1, 1664, -1, 1536}, {7168, 9216, -1, 2147483647, -1, 1536}}},
    {10,
        {{-1, 7168, 1664, 2560, -1, 1536}, {-1, 7168, 5632, 2147483647, -1, 1536},
        {3072, 7168, -1, 2147483647, 1536, 2304}}},
    {6,
        {{-1, 7168, 2560, 5632, -1, 1536}, {-1, 3072, -1, 2147483647, 1536, 2304},
        {9216, 2147483647, 2816, 2147483647, -1, 1536}, {7168, 19456, -1, 2147483647, 2304, 3328},
        {19456, 2147483647, 1664, 2147483647, 1536, 3328}, {19456, 2147483647, -1, 2147483647, 3840, 5632}}},
    {5,
        {{-1, 7168, -1, 3584, 2304, 2816}, {7168, 19456, -1, 2147483647, 1536, 2304},
        {7168, 19456, -1, 4608, 3328, 5632}, {11264, 2147483647, -1, 2304, 5632, 6656},
        {7168, 2147483647, -1, 2304, 6656, 8704}, {9216, 15360, 7680, 2147483647, 6656, 7680}}},
    {3,
        {{-1, 7168, 3584, 2147483647, 2304, 2816}, {-1, 3072, -1, 3584, 2816, 3840},
        {3072, 7168, -1, 2147483647, 2816, 3840}, {-1, 7168, -1, 4608, 3840, 4608},
        {-1, 7168, -1, 3584, 4608, 6656}, {7168, 19456, 4608, 2147483647, 3328, 5632},
        {19456, 2147483647, -1, 2147483647, 3328, 3840}, {9216, 11264, -1, 2147483647, 5632, 6656},
        {-1, 5120, 1152, 1664, 7680, 2147483647}, {5120, 7168, -1, 2304, 6656, 8704},
        {5120, 2147483647, 4608, 5632, 8704, 2147483647}, {9216, 11264, 5632, 6656, 6656, 2147483647}}},
    {1,
        {{-1, 3072, 3584, 2147483647, 2816, 3840}, {-1, 7168, 4608, 2147483647, 3840, 4608},
        {-1, 7168, 3584, 2147483647, 4608, 6656}, {-1, 5120, 2560, 5632, 6656, 7680},
        {-1, 5120, -1, 1152, 7680, 2147483647}, {-1, 5120, 1664, 5632, 7680, 2147483647},
        {5120, 2147483647, 2304, 4608, 9728, 2147483647}, {-1, 3072, 6656, 7680, 6656, 2147483647},
        {-1, 3072, 7680, 2147483647, 6656, 7680}}},
    {12,
        {{9216, 2147483647, -1, 2816, -1, 1536}, {19456, 2147483647, -1, 1664, 1536, 3328}}},
    {2,
        {{7168, 9216, -1, 2147483647, 5632, 6656}, {-1, 5120, -1, 2560, 6656, 7680},
        {5120, 2147483647, -1, 2304, 8704, 2147483647}, {5120, 2147483647, 2304, 4608, 6656, 9728},
        {5120, 2147483647, 4608, 5632, 6656, 8704}, {-1, 3072, 5632, 6656, 6656, 2147483647},
        {-1, 3072, 7680, 2147483647, 7680, 2147483647}, {3072, 9216, 5632, 2147483647, 6656, 2147483647},
        {9216, 2147483647, 7680, 2147483647, 7680, 9728}}},
    {4,
        {{11264, 2147483647, 2304, 2147483647, 5632, 6656}}},
    {8,
        {{11264, 2147483647, 5632, 6656, 6656, 2147483647}, {9216, 2147483647, 6656, 7680, 6656, 2147483647},
        {15360, 2147483647, 7680, 2147483647, 6656, 7680}, {9216, 2147483647, 7680, 2147483647, 9728, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_alltoallmatmulTwoRankFP16FirststepcorenumMap = {
    {16,
        {{-1, 3072, -1, 2688, -1, 1536}, {-1, 3072, -1, 2147483647, 1536, 5632},
        {3072, 2147483647, -1, 2147483647, -1, 5632}, {-1, 11264, -1, 2816, 5632, 2147483647},
        {11264, 2147483647, -1, 2304, 5632, 7680}}},
    {8,
        {{-1, 3072, 2688, 2147483647, -1, 1536}, {11264, 2147483647, -1, 1664, 7680, 2147483647}}},
    {4,
        {{-1, 11264, 2816, 2147483647, 5632, 2147483647}, {11264, 2147483647, 2304, 2147483647, 5632, 7680},
        {11264, 2147483647, 1664, 2147483647, 7680, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_alltoallmatmulFourRankFP16UbsizeMap = {
    {2.0,
        {{-1, 30720, -1, 2816, -1, 2304}, {-1, 30720, 1152, 2816, 2304, 2816},
        {-1, 26624, 1152, 3584, 2816, 3328}, {26624, 30720, 2304, 4352, 2816, 3328},
        {30720, 34816, -1, 1664, -1, 1536}, {30720, 34816, 1664, 2816, -1, 3328},
        {-1, 6144, -1, 1536, 3328, 3840}, {-1, 6144, -1, 1792, 3840, 4608},
        {-1, 6144, -1, 3584, 6656, 7680}}},
    {1.0,
        {{-1, 30720, 2816, 2147483647, -1, 2304}, {-1, 30720, -1, 1152, 2304, 2816},
        {-1, 30720, 2816, 2147483647, 2304, 2816}, {-1, 26624, -1, 1152, 2816, 3328},
        {-1, 26624, 3584, 2147483647, 2816, 3328}, {26624, 30720, -1, 2304, 2816, 3328},
        {26624, 30720, 4352, 2147483647, 2816, 3328}, {30720, 34816, -1, 1664, 1536, 3328},
        {30720, 34816, 2816, 2147483647, -1, 3328}, {34816, 2147483647, -1, 2147483647, -1, 3328},
        {-1, 6144, 1536, 2147483647, 3328, 3840}, {-1, 6144, 1792, 2147483647, 3840, 4608},
        {-1, 6144, -1, 2147483647, 4608, 6656}, {-1, 6144, 3584, 2147483647, 6656, 7680},
        {-1, 6144, -1, 2147483647, 7680, 2147483647}, {6144, 2147483647, -1, 2147483647, 3328, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_alltoallmatmulFourRankFP16M0Map = {
    {256,
        {{-1, 6144, -1, 2688, -1, 1536}, {-1, 6144, 5120, 2147483647, -1, 1536},
        {6144, 18432, 1664, 2147483647, -1, 1536}, {18432, 2147483647, -1, 2147483647, -1, 1536},
        {-1, 6144, 1664, 3584, 5632, 2147483647}, {-1, 6144, 3584, 2147483647, 1536, 2816},
        {-1, 6144, 3584, 2147483647, 3328, 2147483647}, {6144, 2147483647, -1, 7680, 1536, 9728},
        {6144, 2147483647, -1, 4608, 9728, 2147483647}, {6144, 2147483647, 7680, 2147483647, 1536, 2147483647}}},
    {128,
        {{-1, 6144, 2688, 5120, -1, 1536}, {6144, 18432, -1, 1664, -1, 1536},
        {-1, 6144, -1, 3584, 1536, 5632}, {-1, 6144, -1, 1664, 5632, 2147483647},
        {-1, 6144, 3584, 2147483647, 2816, 3328}, {6144, 2147483647, 4608, 7680, 9728, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_alltoallmatmulFourRankFP16FirststepcorenumMap = {
    {4,
        {{-1, 6144, -1, 5120, -1, 1536}, {6144, 14336, -1, 2560, -1, 1536},
        {-1, 14336, -1, 3584, 1536, 2304}, {-1, 10240, -1, 2147483647, 2304, 2816},
        {-1, 6144, -1, 3328, 2816, 3328}, {-1, 6144, 5120, 2147483647, 2816, 3328},
        {-1, 6144, -1, 2147483647, 3328, 4608}, {6144, 2147483647, 2304, 2147483647, 2816, 4608},
        {-1, 6144, -1, 2304, 5632, 8704}, {6144, 14336, 2304, 2147483647, 4608, 9728},
        {14336, 2147483647, 2304, 6656, 4608, 9728}, {-1, 34816, 2304, 5632, 9728, 2147483647},
        {34816, 2147483647, 2304, 6144, 9728, 2147483647}}},
    {8,
        {{-1, 6144, 5120, 2147483647, -1, 1536}, {6144, 14336, 2560, 2147483647, -1, 1536},
        {10240, 14336, -1, 3584, 2304, 2816}, {-1, 10240, 3584, 2147483647, 1536, 2304},
        {10240, 14336, 3584, 2147483647, 1536, 2816}, {14336, 2147483647, -1, 2147483647, -1, 2816},
        {-1, 6144, 3328, 5120, 2816, 3328}, {6144, 2147483647, -1, 2304, 2816, 4608},
        {-1, 2147483647, -1, 2304, 4608, 5632}, {6144, 2147483647, -1, 2304, 5632, 8704},
        {-1, 2147483647, -1, 2304, 8704, 2147483647}, {-1, 6144, 2304, 2147483647, 4608, 9728},
        {14336, 2147483647, 6656, 2147483647, 4608, 9728}, {-1, 34816, 5632, 2147483647, 9728, 2147483647},
        {34816, 2147483647, 6144, 2147483647, 9728, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_alltoallmatmulFourRankFP16PvalueMap = {
    {6,
        {{-1, 6144, -1, 5120, -1, 1536}, {6144, 10240, -1, 4096, -1, 1536},
        {30720, 38912, 2304, 2816, 5632, 9728}, {28672, 2147483647, 1664, 2304, 9728, 2147483647}}},
    {3,
        {{-1, 6144, 5120, 2147483647, -1, 1536}, {-1, 10240, -1, 2304, 1536, 4608},
        {10240, 14336, 1664, 2147483647, 2816, 5632}, {14336, 2147483647, -1, 3584, -1, 3328},
        {14336, 2147483647, 3584, 4608, -1, 1536}, {14336, 18432, 4608, 2147483647, 2816, 3840},
        {14336, 2147483647, 4608, 2147483647, 3840, 5632}, {6144, 26624, 2816, 2147483647, 5632, 9728},
        {6144, 2147483647, 2304, 5632, 9728, 2147483647}, {6144, 2147483647, 7680, 2147483647, 9728, 2147483647}}},
    {2,
        {{6144, 10240, 4096, 2147483647, -1, 1536}, {10240, 14336, -1, 1664, 4608, 5632},
        {14336, 2147483647, -1, 3584, 3328, 4608}, {14336, 2147483647, 3584, 4608, 2304, 4608},
        {-1, 2147483647, -1, 2304, 5632, 6656}, {6144, 30720, 2304, 2816, 5632, 9728},
        {38912, 2147483647, 2304, 2816, 5632, 9728}, {-1, 6144, 4608, 2147483647, 5632, 9728},
        {22528, 28672, 1664, 2304, 9728, 2147483647}, {-1, 6144, 2304, 2147483647, 9728, 2147483647}}},
    {5,
        {{10240, 14336, -1, 7680, -1, 1536}, {10240, 14336, -1, 1664, 1536, 4608}}},
    {1,
        {{10240, 14336, 7680, 2147483647, -1, 1536}, {-1, 10240, -1, 2304, 4608, 5632},
        {-1, 10240, 2304, 2147483647, 1536, 5632}, {10240, 14336, 1664, 2147483647, 1536, 2816},
        {14336, 2147483647, -1, 3584, 4608, 5632}, {14336, 2147483647, 3584, 4608, 1536, 2304},
        {14336, 2147483647, 4608, 2147483647, -1, 2816}, {-1, 2147483647, -1, 2304, 6656, 9728},
        {-1, 6144, 2304, 4608, 5632, 9728}, {-1, 22528, -1, 2304, 9728, 2147483647},
        {22528, 2147483647, -1, 1664, 9728, 2147483647}}},
    {4,
        {{14336, 2147483647, 3584, 4608, 4608, 5632}, {18432, 2147483647, 4608, 2147483647, 2816, 3840},
        {26624, 2147483647, 2816, 4608, 5632, 9728}}},
    {12,
        {{26624, 2147483647, 4608, 2147483647, 5632, 9728}, {6144, 2147483647, 5632, 7680, 9728, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_alltoallmatmulEightRankFP16UbsizeMap = {
    {2.0,
        {{-1, 12288, -1, 3072, -1, 1536}, {12288, 28672, -1, 2816, -1, 1536},
        {-1, 20480, -1, 2816, 1536, 2304}, {28672, 2147483647, -1, 2816, -1, 2304},
        {-1, 2147483647, -1, 2816, 2304, 2147483647}}},
    {1.0,
        {{-1, 12288, 3072, 2147483647, -1, 1536}, {12288, 28672, 2816, 2147483647, -1, 1536},
        {-1, 20480, 2816, 2147483647, 1536, 2304}, {20480, 28672, -1, 2147483647, 1536, 2304},
        {28672, 2147483647, 2816, 2147483647, -1, 2304}, {-1, 2147483647, 2816, 2147483647, 2304, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_alltoallmatmulEightRankFP16M0Map = {
    {256,
        {{-1, 61440, -1, 2147483647, -1, 5632}, {61440, 2147483647, -1, 4608, -1, 5632},
        {61440, 2147483647, 4608, 2147483647, -1, 1536}, {-1, 2147483647, -1, 4608, 5632, 8704},
        {-1, 45056, -1, 4608, 8704, 2147483647}, {45056, 2147483647, -1, 2304, 8704, 2147483647}}},
    {128,
        {{61440, 2147483647, 4608, 2147483647, 1536, 5632}, {-1, 2147483647, 4608, 2147483647, 5632, 8704},
        {-1, 45056, 4608, 2147483647, 8704, 2147483647}, {45056, 2147483647, 2304, 2147483647, 8704, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_alltoallmatmulEightRankFP16PvalueMap = {
    {2,
        {{-1, 12288, -1, 2147483647, -1, 1536}, {12288, 20480, 3328, 2147483647, -1, 1536},
        {20480, 28672, 1536, 2147483647, -1, 1536}, {-1, 20480, -1, 1152, 1536, 5632},
        {-1, 12288, 4608, 2147483647, 1536, 5632}, {28672, 61440, 1664, 2304, -1, 1536},
        {28672, 2147483647, 2304, 2147483647, -1, 1536}, {-1, 12288, -1, 3840, 6656, 7680},
        {-1, 12288, -1, 2304, 7680, 8704}, {-1, 12288, 1152, 2147483647, 8704, 2147483647}}},
    {3,
        {{12288, 20480, -1, 3328, -1, 1536}, {12288, 28672, 2816, 2147483647, 2816, 5632},
        {28672, 49152, -1, 1152, -1, 1536}, {28672, 65536, 1152, 1664, -1, 1536},
        {28672, 53248, -1, 2147483647, 2816, 5632}, {53248, 61440, 3584, 2147483647, 1536, 5632},
        {61440, 2147483647, -1, 4608, 2816, 5632}, {12288, 20480, -1, 1152, 8704, 2147483647},
        {20480, 36864, -1, 1152, 5632, 2147483647}, {12288, 36864, 3584, 4608, 5632, 2147483647},
        {36864, 45056, -1, 4608, 5632, 6656}, {45056, 61440, -1, 2147483647, 5632, 6656},
        {61440, 2147483647, -1, 4608, 5632, 6656}}},
    {5,
        {{20480, 28672, -1, 1536, -1, 1536}, {49152, 2147483647, -1, 1152, -1, 1536},
        {61440, 2147483647, 1664, 2304, -1, 1536}, {61440, 2147483647, 4608, 2147483647, 2816, 6656}}},
    {1,
        {{-1, 20480, 1152, 2816, 1536, 5632}, {20480, 28672, -1, 2816, 1536, 5632},
        {-1, 12288, 2816, 4608, 1536, 5632}, {12288, 28672, 2816, 2147483647, 1536, 2816},
        {28672, 53248, -1, 2147483647, 1536, 2816}, {53248, 61440, -1, 3584, 1536, 5632},
        {61440, 2147483647, -1, 2147483647, 1536, 2816}, {-1, 12288, -1, 3328, 5632, 6656},
        {-1, 12288, -1, 1152, 8704, 2147483647}, {12288, 20480, -1, 1152, 5632, 8704}}},
    {4,
        {{65536, 2147483647, 1152, 1664, -1, 1536}, {12288, 36864, 1152, 3584, 5632, 2147483647}}},
    {6,
        {{-1, 12288, 3328, 2147483647, 5632, 6656}, {-1, 12288, 3840, 2147483647, 6656, 7680},
        {12288, 36864, 4608, 2147483647, 5632, 6656}, {36864, 2147483647, -1, 2304, 6656, 2147483647},
        {77824, 2147483647, 4608, 2147483647, 9216, 2147483647}}},
    {10,
        {{-1, 12288, 2304, 2147483647, 7680, 8704}, {12288, 36864, 4608, 2147483647, 6656, 2147483647},
        {36864, 45056, 4608, 2147483647, 5632, 6656}, {36864, 77824, 4608, 2147483647, 6656, 2147483647},
        {77824, 2147483647, 4608, 2147483647, 6656, 9216}}},
    {12,
        {{36864, 2147483647, 2304, 4608, 6656, 2147483647}}}
};

bool AlltoAllMatmulTiling910::IsCapable()
{
    OP_LOGI(opName_, "Start with AllToAllMatmul tiling.");
    return true;
}

/**
 * @brief 校验attrs信息
 * @return ge::graphStatus
 */
ge::graphStatus AlltoAllMatmulTiling910::CheckAndSetAttrsInfo(AlltoAllMatmulInfo &info)
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(opName_, "Failed to get attrs."), return ge::GRAPH_FAILED);

    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    
    // 判断为空或者空字符串
    OP_TILING_CHECK(group == nullptr, OP_LOGE(opName_, "The input attr group is null pointer."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(group[0] == '\0', OP_LOGE(opName_, "The input attr group is empty string."),
                    return ge::GRAPH_FAILED);
    info.worldSize = mc2tiling::MatmulFormulaicTiling::GetRankSize(group);
    worldSize = info.worldSize;
    OP_TILING_CHECK(
        SUPPORT_RANK_SIZE_910.find(info.worldSize) == SUPPORT_RANK_SIZE_910.end(),
        OP_LOGE(opName_, "World_size should be 2 or 4 or 8, but the actual value is %u.", info.worldSize),
        return ge::GRAPH_FAILED);

    const bool *isTransX1 = attrs->GetAttrPointer<bool>(ALLTOALLMATMUL_ATTR_X1_TRANSPOSE_INDEX);
    bool x1TransposeFlag = (isTransX1 != nullptr) ? *isTransX1 : false;
    OP_TILING_CHECK(x1TransposeFlag, OP_LOGE(opName_, "X1 transpose is not supported, should be false."),
                    return ge::GRAPH_FAILED);
    
    const bool *isTransX2 = attrs->GetAttrPointer<bool>(ALLTOALLMATMUL_ATTR_X2_TRANSPOSE_INDEX);
    bool x2TransposeFlag = (isTransX2 != nullptr) ? *isTransX2 : false;
    OP_TILING_CHECK(!x2TransposeFlag, OP_LOGE(opName_, "X2 transpose should be true."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 非量化场景校验参数的DType
 * @return ge::graphStatus
 */
ge::graphStatus AlltoAllMatmulTiling910::CheckTensorDataType(AlltoAllMatmulInfo &info)
{
    // 获取并校验输入张量描述符
    auto x1TensorDesc = context_->GetInputDesc(INPUT_X1_INDEX);
    OP_TILING_CHECK((x1TensorDesc == nullptr), OP_LOGE(opName_, "The input tensor x1 is invalid."),
                    return ge::GRAPH_FAILED);
    auto x2TensorDesc = context_->GetInputDesc(INPUT_X2_INDEX);
    OP_TILING_CHECK((x2TensorDesc == nullptr), OP_LOGE(opName_, "The input tensor x2 is invalid."),
                    return ge::GRAPH_FAILED);
    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE(opName_, "Output tensor y is nullptr."), return ge::GRAPH_FAILED);

    // 获取数据类型并校验一致性与范围
    ge::DataType x1Dtype = x1TensorDesc->GetDataType();
    ge::DataType x2Dtype = x2TensorDesc->GetDataType();
    ge::DataType yDtype = yDesc->GetDataType();
    auto biasTensorDesc = context_->GetOptionalInputDesc(INPUT_BIAS_INDEX);

    if (x2Dtype == ge::DT_INT8) {
        // 校验 scale 张量不为空，量化模式
        auto x1ScaleTensorDesc = context_->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
        auto x2ScaleTensorDesc = context_->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
        OP_TILING_CHECK((x2ScaleTensorDesc == nullptr || biasTensorDesc == nullptr),
                        OP_LOGE(opName_, "x2Scale and bias tensors should not be null in quant mode."), return ge::GRAPH_FAILED);
        ge::DataType x2ScaleDtype = x2ScaleTensorDesc->GetDataType();
        OP_TILING_CHECK(x2ScaleDtype != ge::DT_FLOAT,
                        OP_LOGE(opName_, "Scale tensors Dtype should be FLOAT, but x2Scale Dtype is %s.", Ops::Base::ToString(x2ScaleDtype).c_str()),
                        return ge::GRAPH_FAILED);
        isQuant = true;
    }

    // 校验 bias 数据类型（如果存在）
    if (biasTensorDesc != nullptr) {
        hasBias = true;
        ge::DataType biasDtype = biasTensorDesc->GetDataType();
        vector<uint32_t> paramsType = {x1Dtype, x2Dtype, biasDtype, yDtype};

        for (uint32_t kindsId = 0; kindsId < SUPPORTED_TYPES_WITH_BIAS.size(); kindsId++) {
            if (IsArrayEqual(paramsType, SUPPORTED_TYPES_WITH_BIAS[kindsId], COUNT_PARAMS_WITH_BIAS)) {
                return ge::GRAPH_SUCCESS;
            }
        }
        OP_LOGE(opName_,
                "AllToAllMatmul: unSupported params data type [x1, x2, bias, y]: [%s, %s, %s, %s].",
                Ops::Base::ToString(x1Dtype).c_str(),
                Ops::Base::ToString(x2Dtype).c_str(),
                Ops::Base::ToString(biasDtype).c_str(),
                Ops::Base::ToString(yDtype).c_str());
        return ge::GRAPH_FAILED;
    } else {
        vector<uint32_t> paramsType = {x1Dtype, x2Dtype, yDtype};

        for (uint32_t kindsId = 0; kindsId < SUPPORTED_TYPES_WITHOUT_BIAS.size(); kindsId++) {
            if (IsArrayEqual(paramsType, SUPPORTED_TYPES_WITHOUT_BIAS[kindsId], COUNT_PARAMS_WITHOUT_BIAS)) {
                return ge::GRAPH_SUCCESS;
            }
        }
        OP_LOGE(opName_,
                "AllToAllMatmul: unSupported params data type [x1, x2, y]: [%s, %s, %s].",
                Ops::Base::ToString(x1Dtype).c_str(),
                Ops::Base::ToString(x2Dtype).c_str(),
                Ops::Base::ToString(yDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling输入的shape信息
 * @return ge::graphStatus
 */
ge::graphStatus AlltoAllMatmulTiling910::CheckShapeInfo(AlltoAllMatmulInfo &info)
{
    ge::graphStatus status;

    // 校验输入Input Shape是否为空
    const gert::StorageShape *x1Shape = context_->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context_->GetInputShape(INPUT_X2_INDEX);
    OP_TILING_CHECK((x1Shape == nullptr) || (x2Shape == nullptr), OP_LOGE(opName_, "The input shape is invalid"),
                    return ge::GRAPH_FAILED);

    // 校验维度数目是否合法
    uint64_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    uint64_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((x1DimNum != 2 || x2DimNum != 2), 
                    OP_LOGE(opName_, "The input dimNum should be two, but x1DimNum is %lu, x2DimNum is %lu.", x1DimNum, x2DimNum),
                    return ge::GRAPH_FAILED);

    // 校验输出
    const gert::StorageShape *yShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yShape == nullptr), OP_LOGE(opName_, "The yShape is nullptr."), return ge::GRAPH_FAILED);
    uint64_t yDimNum = yShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((yDimNum != 2), 
                    OP_LOGE(opName_, "The output dimNum should be two, but yDimNum is %lu.", yDimNum),
                    return ge::GRAPH_FAILED);

    // 校验shape的维度矩阵是否合法
    status = CheckMatrixMulShapes(context_, opName_);
    if (status != ge::GRAPH_SUCCESS)
        return status;

    // 校验量化场景中scale的shape信息
    auto x1TensorDesc = context_->GetInputDesc(INPUT_X1_INDEX);
    ge::DataType x1Dtype = x1TensorDesc->GetDataType();
    info.M = x1Shape->GetStorageShape().GetDim(0);
    info.K = x1Shape->GetStorageShape().GetDim(1);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);
    info.N = (info.K * info.worldSize == x2Dim1) ? x2Dim0 : x2Dim1;
    if (isQuant) {
        orgM = info.M;
        orgN = info.N;
        const gert::StorageShape *x2ScaleShape = context_->GetOptionalInputShape(INPUT_X2_SCALE_INDEX);
        uint64_t x2ScaleShapeDimNum = x2ScaleShape->GetStorageShape().GetDimNum();
        uint64_t x2ScaleDim0 = x2ScaleShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK((x2ScaleDim0 != info.N),
                        OP_LOGE(opName_, "The x2Scale dimNum0 should be %u, but actual value is %lu.", info.N, x2ScaleDim0),
                        return ge::GRAPH_FAILED);
    }
    const gert::StorageShape *biasShape = context_->GetOptionalInputShape(INPUT_BIAS_INDEX);
    if (biasShape != nullptr) {
        uint64_t biasShapeDimNum = biasShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK((biasShapeDimNum != 1), OP_LOGE(opName_, "The input bias dimNum should be one."),
                        return ge::GRAPH_FAILED);
        uint64_t biasDim0 = biasShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK((biasDim0 != info.N),
                        OP_LOGE(opName_, "The bias dimNum0 should be %u, but actual value is %lu.", info.N, biasDim0),
                        return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(info.M == 0, OP_LOGE(opName_, "Invalid x1 shape: dim 0(m) cannot be 0."), return ge::GRAPH_FAILED);
    // 校验K,K的范围应该在[1, 65535]
    OP_TILING_CHECK(info.K > K_MAX_VALUE, OP_LOGE(opName_, "X1 dim 1(k) exceeds max value 65535, got %u.", info.K),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(info.N == 0, OP_LOGE(opName_, "Invalid x2 shape: N cannot be 0."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验输入信息是否合规:attr,Dtype,shape等，使用通用校验util中的check方法
 *
 * @return ge::graphStatus
 */
ge::graphStatus AlltoAllMatmulTiling910::CheckOpInputInfo(AlltoAllMatmulInfo &info)
{
    OP_TILING_CHECK(CheckAndSetAttrsInfo(info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorDataType(info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Dtype failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckShapeInfo(info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check shape failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

int32_t AlltoAllMatmulTiling910::GetValueFromMKNConditionMap(int32_t m, int32_t k, int32_t n, int32_t defaultValue, 
                                    std::map<int, std::vector<std::vector<int>>> conditionMap)
{
    int32_t value = defaultValue;
    for (auto &item : conditionMap) {
        for (auto &condition : item.second) {
            bool inRange =
                m > condition[CONDITION_M_ST] && m <= condition[CONDITION_M_END] &&
                k > condition[CONDITION_K_ST] && k <= condition[CONDITION_K_END] &&
                n > condition[CONDITION_N_ST] && n <= condition[CONDITION_N_END];
            if (inRange) {
                return item.first;
            }
        }
    }
    return value;
}

void AlltoAllMatmulTiling910::CalTilingParam(CoCTiling &cocTilingData, const std::map<int*, AlltoAllMatmulTilingValue>& TilingParamMap, AlltoAllMatmulInfo &info)
{
    int32_t m = info.M;
    int32_t k = info.K;
    int32_t n = info.N;

    for (auto &item : TilingParamMap) {
        auto value = item.second.value;
        auto conditionMap = item.second.conditionMap;
        if (!conditionMap.empty()) {
            *item.first = GetValueFromMKNConditionMap(m, k, n, value, conditionMap);
        } else if (value != -1) {
            *item.first = value;
        }
    }
    cocTilingData.ubMoveNum = cocTilingData.ubMoveNum * HALF_KBYTE;
    if (cocTilingData.m0 >= DEFAULT_ROW) {
        cocTilingData.k0 = DEFAULT_COL;
        cocTilingData.n0 = cocTilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;
    }
}

void TilingParamDeal(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info, int32_t ubSize)
{
    uint32_t k = info.K;
    uint32_t rankSize = info.worldSize;
    int32_t dataTypeSize = ELEMENT_SIZE;
    int32_t peerMemSize = (MAX_BUFF_BYTES - FLAG_BUFF_BYTES) / dataTypeSize;
    int32_t transKAlign = RoundNum(k * rankSize, HALF_KBYTE / dataTypeSize);
    cocTilingData.ubMoveNum = std::min(ubSize * transKAlign * UB_PINGPONG_SIZE, MAX_UB_NUM);
    if (cocTilingData.m0 * cocTilingData.pValue * transKAlign > peerMemSize / MAX_BLOCK_COUNT) {
        int32_t maxValue = peerMemSize / MAX_BLOCK_COUNT / transKAlign;
        cocTilingData.m0 = DEFAULT_ROW;
        cocTilingData.n0 = cocTilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;
        cocTilingData.pValue = ClampValue(maxValue / cocTilingData.m0, MIN_P_VALUE, cocTilingData.pValue);
    }
}

void AlltoAllMatmulTiling910::DoTwoRankTiling(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info)
{
    int32_t ubSize = ALLTOALLMATMUL_TWO_RANK_FP16_UBSIZE_DEFAULT;
    std::map<int*, AlltoAllMatmulTilingValue> TilingParamMap;
    TilingParamMap[&cocTilingData.m0] = AlltoAllMatmulTilingValue(ALLTOALLMATMUL_TWO_RANK_FP16_M0_DEFAULT, g_alltoallmatmulTwoRankFP16M0Map);
    TilingParamMap[&cocTilingData.pValue] = AlltoAllMatmulTilingValue(ALLTOALLMATMUL_TWO_RANK_FP16_PVALUE_DEFAULT, g_alltoallmatmulTwoRankFP16PvalueMap);
    TilingParamMap[&ubSize] = AlltoAllMatmulTilingValue(ALLTOALLMATMUL_TWO_RANK_FP16_UBSIZE_DEFAULT, g_alltoallmatmulTwoRankFP16UbsizeMap);
    TilingParamMap[&cocTilingData.swizzlDirect] = AlltoAllMatmulTilingValue(SWIZZLE_DIRECT_ONE);
    TilingParamMap[&cocTilingData.swizzlCount] = AlltoAllMatmulTilingValue(DEFAULT_SWIZZLE_COUNT);
    TilingParamMap[&cocTilingData.first_step_core_num] = AlltoAllMatmulTilingValue(ALLTOALLMATMUL_TWO_RANK_FP16_FIRSTSTEPCORENUM_DEFAULT,
        g_alltoallmatmulTwoRankFP16FirststepcorenumMap);
    TilingParamMap[&cocTilingData.second_step_core_num] = AlltoAllMatmulTilingValue(CORE_NUM_FOUR);
    CalTilingParam(cocTilingData, TilingParamMap, info);
    TilingParamDeal(cocTilingData, info, ubSize);
}

void AlltoAllMatmulTiling910::DoFourRankTiling(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info)
{
    int32_t ubSize = ALLTOALLMATMUL_FOUR_RANK_FP16_UBSIZE_DEFAULT;
    std::map<int*, AlltoAllMatmulTilingValue> TilingParamMap;
    TilingParamMap[&cocTilingData.m0] = AlltoAllMatmulTilingValue(ALLTOALLMATMUL_FOUR_RANK_FP16_M0_DEFAULT, g_alltoallmatmulFourRankFP16M0Map);
    TilingParamMap[&cocTilingData.pValue] = AlltoAllMatmulTilingValue(ALLTOALLMATMUL_FOUR_RANK_FP16_PVALUE_DEFAULT, g_alltoallmatmulFourRankFP16PvalueMap);
    TilingParamMap[&ubSize] = AlltoAllMatmulTilingValue(ALLTOALLMATMUL_FOUR_RANK_FP16_UBSIZE_DEFAULT, g_alltoallmatmulFourRankFP16UbsizeMap);
    TilingParamMap[&cocTilingData.swizzlDirect] = AlltoAllMatmulTilingValue(SWIZZLE_DIRECT_ONE);
    TilingParamMap[&cocTilingData.swizzlCount] = AlltoAllMatmulTilingValue(SWIZZLE_COUNT_THREE);
    TilingParamMap[&cocTilingData.first_step_core_num] = AlltoAllMatmulTilingValue(ALLTOALLMATMUL_FOUR_RANK_FP16_FIRSTSTEPCORENUM_DEFAULT,
        g_alltoallmatmulFourRankFP16FirststepcorenumMap);
    TilingParamMap[&cocTilingData.second_step_core_num] = AlltoAllMatmulTilingValue(CORE_NUM_EIGHT);
    CalTilingParam(cocTilingData, TilingParamMap, info);
    TilingParamDeal(cocTilingData, info, ubSize);
}

void AlltoAllMatmulTiling910::DoEightRankTiling(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info)
{
    int32_t ubSize = ALLTOALLMATMUL_EIGHT_RANK_FP16_UBSIZE_DEFAULT;
    std::map<int*, AlltoAllMatmulTilingValue> TilingParamMap;
    TilingParamMap[&cocTilingData.m0] = AlltoAllMatmulTilingValue(ALLTOALLMATMUL_EIGHT_RANK_FP16_M0_DEFAULT, g_alltoallmatmulEightRankFP16M0Map);
    TilingParamMap[&cocTilingData.pValue] = AlltoAllMatmulTilingValue(ALLTOALLMATMUL_EIGHT_RANK_FP16_PVALUE_DEFAULT, g_alltoallmatmulEightRankFP16PvalueMap);
    TilingParamMap[&ubSize] = AlltoAllMatmulTilingValue(ALLTOALLMATMUL_EIGHT_RANK_FP16_UBSIZE_DEFAULT, g_alltoallmatmulEightRankFP16UbsizeMap);
    TilingParamMap[&cocTilingData.swizzlDirect] = AlltoAllMatmulTilingValue(SWIZZLE_DIRECT_ONE);
    TilingParamMap[&cocTilingData.swizzlCount] = AlltoAllMatmulTilingValue(SWIZZLE_COUNT_THREE);
    TilingParamMap[&cocTilingData.first_step_core_num] = AlltoAllMatmulTilingValue(CORE_NUM_EIGHT);
    TilingParamMap[&cocTilingData.second_step_core_num] = AlltoAllMatmulTilingValue(CORE_NUM_EIGHT);
    CalTilingParam(cocTilingData, TilingParamMap, info);
    TilingParamDeal(cocTilingData, info, ubSize);
}

ge::graphStatus AlltoAllMatmulTiling910::DoMmCommTiling(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info)
{
    if (info.worldSize == WORLDSIZE_TWO) {
        DoTwoRankTiling(cocTilingData, info);
        return ge::GRAPH_SUCCESS;
    } else if (info.worldSize == WORLDSIZE_FOUR) {
        DoFourRankTiling(cocTilingData, info);
        return ge::GRAPH_SUCCESS;
    }
    DoEightRankTiling(cocTilingData, info);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllMatmulTiling910::DoOpTiling()
{
    // 1. tilingData
    AlltoAllMatmulTilingData *tilingData = context_->GetTilingData<AlltoAllMatmulTilingData>();
    OPS_CHECK(tilingData == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);
    OPS_LOG_I(opName_, "AllToAllMatmul get tilingData.");
    AlltoAllMatmulInfo& info = tilingData->allToAllMatmulInfo;
    OPS_LOG_I(opName_, "AllToAllMatmul get tilingData info.");

    GE_ASSERT_GRAPH_SUCCESS(CheckOpInputInfo(info));
    GE_ASSERT_GRAPH_SUCCESS(DoMmCommTiling(tilingData->cocTiling, info));
    GE_ASSERT_GRAPH_SUCCESS(SetHcclTiling(tilingData));
    SetTilingKey(info);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);

    OPS_LOG_I(opName_, "Leave AllToAllMatmul tiling func.");
    return ge::GRAPH_SUCCESS;
}

void AlltoAllMatmulTiling910::SetTilingKey(AlltoAllMatmulInfo &info)
{
    tilingKey_ = INIT_TILINGKEY;
    tilingKey_ += hasBias ? TILINGKEY_BIAS : 0;
    OP_LOGD(opName_, "TilingKey is [%lu] in AllToAllMatmul.", tilingKey_);
}

/**
 * @brief 获取对应的tilingKey
 *
 * @return uint64_t tilingKey结果
 */
uint64_t AlltoAllMatmulTiling910::GetTilingKey() const
{
    return tilingKey_;
}

/**
 * @brief 设置hccl的config,进行hccl对应的通信任务设置
 *
 * @return ge::graphStatus
 */
ge::graphStatus AlltoAllMatmulTiling910::SetHcclTiling(AlltoAllMatmulTilingData *tilingData)
{
    auto attrs = context_->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    uint32_t opType = 18; // batch write=18,
    std::string algConfig = "MultiPut=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取额外申请的空间
 *
 * @return ge::graphStatus
 */
ge::graphStatus AlltoAllMatmulTiling910::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "Get workspace failed"), return ge::GRAPH_FAILED);
    size_t wsSize = SYSTEM_NEED_WORKSPACE;
    if (isQuant) {
        wsSize += orgM * orgN * sizeof(int32_t) / worldSize;
    }
    workspaces[0] = wsSize;
    OP_LOGD(opName_, "Workspaces[0] size=%ld", workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 打印tilingInfo信息
 *
 * @param opName_
 * @param tilingInfo
 */
void AlltoAllMatmulTiling910::PrintAlltoAllMatmulTilingData(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info)
{
    OP_LOGD(opName_, "info.M: %u", info.M);
    OP_LOGD(opName_, "info.K: %u", info.K);
    OP_LOGD(opName_, "info.N: %u", info.N);
    OP_LOGD(opName_, "info.worldSize: %u", info.worldSize);
    OP_LOGD(opName_, "info.aivNum: %u", info.aivNum);
    OP_LOGD(opName_, "info.totalUbSize: %u", info.totalUbSize);
    OP_LOGD(opName_, "info.hasBias: %d", info.hasBias);
    OP_LOGD(opName_, "cocTilingData.m0: %u", cocTilingData.m0);
    OP_LOGD(opName_, "cocTilingData.pValue: %u", cocTilingData.pValue);
    OP_LOGD(opName_, "cocTilingData.unMoveNum: %u", cocTilingData.ubMoveNum);
    OP_LOGD(opName_, "cocTilingData.swizzlDirect: %u", cocTilingData.swizzlDirect);
    OP_LOGD(opName_, "cocTilingData.swizzlCount: %u", cocTilingData.swizzlCount);
    OP_LOGD(opName_, "cocTilingData.first_step_core_num: %u", cocTilingData.first_step_core_num);
    OP_LOGD(opName_, "cocTilingData.second_step_core_num: %u", cocTilingData.second_step_core_num);
}

/**
 * @brief 保存tiling数据到context
 *
 * @return ge::graphStatus
 */
ge::graphStatus AlltoAllMatmulTiling910::PostTiling()
{
    AlltoAllMatmulTilingData *outTilingData = context_->GetTilingData<AlltoAllMatmulTilingData>();
    context_->SetBlockDim(blockDim);

    PrintAlltoAllMatmulTilingData(outTilingData->cocTiling, outTilingData->allToAllMatmulInfo);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 构造函数，创建一个AlltoAllMatmulTiling910对象
 *
 * @param context
 */
AlltoAllMatmulTiling910::AlltoAllMatmulTiling910(gert::TilingContext *context) : AllToAllMatmulTilingBase(context)
{
}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(AlltoAllMatmul, AlltoAllMatmulTiling910,
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910B), 0);
} // namespace MC2Tiling