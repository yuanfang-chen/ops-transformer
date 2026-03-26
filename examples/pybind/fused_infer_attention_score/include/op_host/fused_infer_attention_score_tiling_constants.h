/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling_constants.h
 * \brief
 */

#ifndef FUSED_INFER_ATTENTION_SCORE_TILING_CONSTANTS_H
#define FUSED_INFER_ATTENTION_SCORE_TILING_CONSTANTS_H

// #include "../../common/op_host/fia_tiling_info.h"
#include <map>
#include <set>
#include <vector>
#include <cstring>
#include "fused_infer_attention_score_tiling_index.h"

namespace optiling {
namespace arch35FIA {
constexpr uint64_t BENCHMARK_TILING_KEY_0 = 3000000000000000000;
constexpr uint64_t BENCHMARK_TILING_KEY_1 = 3000000000000000001;
constexpr uint64_t BENCHMARK_TILING_KEY_2 = 3000000000000000002;
constexpr uint64_t BENCHMARK_TILING_KEY_3 = 3000000000000000003;
constexpr uint64_t BENCHMARK_TILING_KEY_4 = 2000000000004000000U;
constexpr uint64_t tiling_key_10000U = 10000U;
constexpr uint64_t tiling_key_20000U = 20000U;
constexpr bool GRAPH_SUCCESS = true;
constexpr bool GRAPH_FAILED = false;
#define OP_LOGE(name, message) \
    do { \
        fprintf(stderr, "[ERROR] [%s] %s\n", (name), (message)); \
    } while(0)

#define OP_CHECK_IF(condition, log_statement, return_statement) \
    do { \
        if (condition) { \
            log_statement; \
            return_statement; \
        } \
    } while(0)
#define OPS_REPORT_VECTOR_INNER_ERR(name, message) \
    OP_LOGE(name, message)
struct DataType {
    int id;
    const char* name;

    // 重载 < 运算符，用于 map 排序
    bool operator<(const DataType& other) const {
        if (id != other.id) return id < other.id;
        return name < other.name;
    }

    // 重载 == 运算符，用于 map 排序
    bool operator==(const DataType& other) const {
        if (id != other.id) return false;
        // 字符串比较：比指针不安全，要比内容
        return std::strcmp(name, other.name) == 0;
    }
    bool operator!=(const DataType& other) const {
        return !(*this == other);
    }
};

struct Format {
    int id;
    const char* name;

    // 重载 < 运算符，用于 map 排序
    bool operator<(const Format& other) const {
        if (id != other.id) return id < other.id;
        return name < other.name;
    }

    // 重载 == ：判断两个 Format 是否相等（比较内容）
    bool operator==(const Format& other) const {
        if (id != other.id) {
            return false;
        }
        // 比较字符串内容，不是比较指针
        return std::strcmp(name, other.name) == 0;
    }

    // 重载 != ：配套使用，更方便
    bool operator!=(const Format& other) const {
        return !(*this == other);
    }
};

constexpr Format FORMAT_ND = {1, "FORMAT_ND"};
constexpr Format FORMAT_NCHW = {2, "FORMAT_NCHW"};
constexpr Format FORMAT_NHWC = {3, "FORMAT_NHWC"};
constexpr Format FORMAT_NCDHW = {4, "FORMAT_NCDHW"};
constexpr DataType DT_UNDEFINED = {1, "DT_UNDEFINED"};
constexpr DataType DT_FLOAT = {2, "DT_FLOAT"};                   // float type
constexpr DataType DT_FLOAT16 = {3, "DT_FLOAT16"};               // fp16 type
constexpr DataType DT_INT8 = {4, "DT_INT8"};                     // int8 type
constexpr DataType DT_INT16 = {5, "DT_INT16"};                   // int16 type
constexpr DataType DT_UINT16 = {6, "DT_UINT16"};                 // uint16 type
constexpr DataType DT_UINT8 = {7, "DT_UINT8"};                   // uint8 type
constexpr DataType DT_INT32 = {8, "DT_INT32"};                   // int32 type
constexpr DataType DT_INT64 = {9, "DT_INT64"};                   // int64 type
constexpr DataType DT_UINT32 = {10, "DT_UINT32"};                 // unsigned int32
constexpr DataType DT_UINT64 = {11, "DT_UINT64"};                 // unsigned int64
constexpr DataType DT_BOOL = {12, "DT_BOOL"};                     // bool type
constexpr DataType DT_DOUBLE = {13, "DT_DOUBLE"};                 // double type
constexpr DataType DT_DUAL = {14, "DT_DUAL"};                     // dual output type
constexpr DataType DT_DUAL_SUB_INT8 = {15, "DT_DUAL_SUB_INT8"};   // dual output int8 type
constexpr DataType DT_DUAL_SUB_UINT8 = {16, "DT_DUAL_SUB_UINT8"}; // dual output uint8 type
constexpr DataType DT_COMPLEX32 = {17, "DT_COMPLEX32"};           // complex32 type
constexpr DataType DT_COMPLEX64 = {18, "DT_COMPLEX64"};           // complex64 type
constexpr DataType DT_COMPLEX128 = {19, "DT_COMPLEX128"};         // complex128 type
constexpr DataType DT_QINT8 = {20, "DT_QINT8"};                   // qint8 type
constexpr DataType DT_QINT16 = {21, "DT_QINT16"};                 // qint16 type
constexpr DataType DT_QINT32 = {22, "DT_QINT32"};                 // qint32 type
constexpr DataType DT_QUINT8 = {23, "DT_QUINT8"};                 // quint8 type
constexpr DataType DT_QUINT16 = {24, "DT_QUINT16"};               // quint16 type
constexpr DataType DT_RESOURCE = {25, "DT_RESOURCE"};             // resource type
constexpr DataType DT_STRING_REF = {26, "DT_STRING_REF"};         // string ref type
constexpr DataType DT_STRING = {27, "DT_STRING"};                 // string type
constexpr DataType DT_VARIANT = {28, "DT_VARIANT"};               // dt_variant type
constexpr DataType DT_BF16 = {29, "DT_BFLOAT16"};                 // dt_bfloat16 type
constexpr DataType DT_INT4 = {30, "DT_INT4"};
constexpr DataType DT_UINT1 = {31, "DT_UINT1"};
constexpr DataType DT_INT2 = {32, "DT_INT2"};
constexpr DataType DT_UINT2 = {33, "DT_UINT2"};
constexpr DataType DT_HIFLOAT8 = {34, "DT_HIFLOAT8"};
constexpr DataType DT_FLOAT8_E4M3FN = {35, "DT_FLOAT8_E4M3FN"};
constexpr DataType DT_FLOAT4_E2M1 = {36, "DT_FLOAT4_E2M1"};
// shape limit
constexpr uint32_t B_LIMIT = 65536;
constexpr uint32_t N1_LIMIT = 256;
constexpr uint32_t N2_LIMIT = 256;
constexpr uint32_t G_LIMIT = 64;
constexpr uint32_t D_LIMIT = 512;
constexpr uint32_t T_LIMIT = 1048576;
constexpr uint32_t H_LIMIT = 65535;
constexpr uint32_t S_LIMIT = 20971520;

constexpr uint32_t INPUT_Q_SHAPE_MIN_DIMS = 3;
constexpr uint32_t INPUT_Q_SHAPE_MAX_DIMS = 4;
constexpr uint32_t INPUT_KV_SHAPE_MIN_DIMS = 3;
constexpr uint32_t INPUT_KV_SHAPE_MAX_DIMS = 5;

constexpr uint32_t DIM_NUM_0 = 0;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_NUM_5 = 5;

// PA
constexpr uint32_t BLOCK_SIZE_MAX = 512;
constexpr uint32_t BLOCK_SIZE_MAX_FOR_NO_QUANT = 1024;
constexpr uint32_t BLOCK_SIZE_ALIGN_SIZE_16 = 16;
constexpr uint32_t BLOCK_SIZE_ALIGN_SIZE_128 = 128;

// sparse mode
// constexpr uint32_t SPARSE_MODE_NO_MASK = 0;
// constexpr uint32_t SPARSE_MODE_ALL_MASK = 1;
// constexpr uint32_t SPARSE_MODE_LEFT_UP = 2;
// constexpr uint32_t SPARSE_MODE_RIGHT_DOWN = 3;
// constexpr uint32_t SPARSE_MODE_BAND = 4;
constexpr uint32_t SPARSE_OPTIMIZE_ATTENTION_SIZE = 2048;

// mask
constexpr uint32_t MASK_DIM_SS = 2;
constexpr uint32_t MASK_DIM_BSS = 3;
constexpr uint32_t MASK_DIM_B1SS = 4;

// dequant mode
constexpr uint32_t PER_CHANNEL_MODE = 0;
constexpr uint32_t PER_TOKEN_MODE = 1;
constexpr uint32_t PER_TENSOR_HEAD_MODE = 2;
constexpr uint32_t PER_TOKEN_HEAD_MODE = 3;
constexpr uint32_t PER_TOKEN_PA_MODE = 4;
constexpr uint32_t PER_TOKEN_HEAD_PA_MODE = 5;
constexpr uint32_t PER_TOKEN_GROUP_MODE = 6;
constexpr uint32_t PER_BLOCK_MODE = 7;

// ROPE MLA
constexpr uint32_t MLA_QKD_SIZE = 192;
constexpr uint32_t MLA_VD_SIZE = 128;
constexpr uint32_t MLA_D_DIM_512 = 512;
constexpr uint32_t MLA_ROPE_D_DIM_64 = 64;

// inner precise
constexpr uint32_t INNER_PRECISE_LIMIT = 4;
constexpr uint32_t HIGH_PRECISION = 0;
constexpr uint32_t HIGH_PERFORMANCE = 1;

constexpr uint32_t BYTE_BLOCK = 32;
constexpr int64_t SHAPE_PARAMS_CONST = 1;
constexpr int64_t SHAPE_NUM_ONE = 1;

// datatype size
constexpr uint32_t FLOAT16SIZE = 2;
constexpr uint32_t BFLOAT16SIZE = 2;
constexpr uint32_t INT8SIZE = 1;
constexpr uint32_t FLOAT8SIZE = 1;
constexpr float INT4SIZE = 0.5f;
constexpr float FLOAT4SIZE = 0.5f;

constexpr uint32_t SOUTER_32 = 32;
constexpr uint32_t SOUTER_64 = 64;
constexpr uint32_t SOUTER_128 = 128;
constexpr uint32_t SINNER_64 = 64;
constexpr uint32_t SINNER_128 = 128;
constexpr uint32_t SINNER_256 = 256;
constexpr uint32_t DSIZE_64 = 64;
constexpr uint32_t DSIZE_128 = 128;
constexpr uint32_t DSIZE_192 = 192;
constexpr uint32_t DSIZE_256 = 256;
constexpr uint32_t DSIZE_512 = 512;
constexpr uint32_t DSIZE_576 = 576;
constexpr uint32_t CV_RATIO = 2;

constexpr uint32_t DOUBLE_BUFFER_NUM=2;

// num
constexpr uint32_t NUM1 = 1;
constexpr uint32_t NUM8 = 8;
constexpr uint32_t NUM_16 = 16;
constexpr uint32_t NUM_32 = 32;
constexpr uint32_t NUM_64 = 64;
constexpr uint32_t NUM_128 = 128;
constexpr uint32_t NUM_192 = 192;
constexpr uint32_t NUM_256 = 256;
constexpr uint32_t NUM_512 = 512;
constexpr uint32_t NUM_1024 = 1024;
constexpr uint32_t NUM_2048 = 2048;

const std::map<DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {DT_FLOAT, "DT_FLOAT"},                   // float type
    {DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {DT_INT8, "DT_INT8"},                     // int8 type
    {DT_INT16, "DT_INT16"},                   // int16 type
    {DT_UINT16, "DT_UINT16"},                 // uint16 type
    {DT_UINT8, "DT_UINT8"},                   // uint8 type
    {DT_INT32, "DT_INT32"},                   // int32 type
    {DT_INT64, "DT_INT64"},                   // int64 type
    {DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {DT_BOOL, "DT_BOOL"},                     // bool type
    {DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {DT_DUAL, "DT_DUAL"},                     // dual output type
    {DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {DT_QINT8, "DT_QINT8"},                   // qint8 type
    {DT_QINT16, "DT_QINT16"},                 // qint16 type
    {DT_QINT32, "DT_QINT32"},                 // qint32 type
    {DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {DT_STRING, "DT_STRING"},                 // string type
    {DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {DT_INT4, "DT_INT4"},
    {DT_UINT1, "DT_UINT1"},
    {DT_INT2, "DT_INT2"},
    {DT_UINT2, "DT_UINT2"},
    {DT_HIFLOAT8, "DT_HIFLOAT8"},
    {DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
    {DT_FLOAT4_E2M1, "DT_FLOAT4_E2M1"}
};

const std::map<std::string, std::vector<DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME,                  {DT_FLOAT16, DT_BF16, DT_INT8}},
    {KEY_NAME,                    {DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}},
    {VALUE_NAME,                  {DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}},
    {PSE_SHIFT_NAME,              {DT_FLOAT16, DT_BF16}},
    {ATTEN_MASK_NAME,             {DT_BOOL, DT_INT8, DT_UINT8}},
    {DEQUANT_SCALE1_NAME,         {DT_UINT64, DT_FLOAT}},
    {QUANT_SCALE1_NAME,           {DT_FLOAT}},
    {DEQUANT_SCALE2_NAME,         {DT_UINT64, DT_FLOAT}},
    {QUANT_SCALE2_NAME,           {DT_FLOAT, DT_BF16}},
    {QUANT_OFFSET2_NAME,          {DT_FLOAT, DT_BF16}},
    {ANTIQUANT_SCALE_NAME,        {DT_FLOAT16, DT_BF16}},
    {ANTIQUANT_OFFSET_NAME,       {DT_FLOAT16, DT_BF16}},
    {BLOCK_TABLE_NAME,            {DT_INT32}},
    {KEY_ANTIQUANT_SCALE_NAME,    {DT_FLOAT16, DT_BF16, DT_FLOAT}},
    {KEY_ANTIQUANT_OFFSET_NAME,   {DT_FLOAT16, DT_BF16, DT_FLOAT}},
    {VALUE_ANTIQUANT_SCALE_NAME,  {DT_FLOAT16, DT_BF16, DT_FLOAT}},
    {VALUE_ANTIQUANT_OFFSET_NAME, {DT_FLOAT16, DT_BF16, DT_FLOAT}},
    {KEY_SHARED_PREFIX_NAME,      {DT_FLOAT16, DT_BF16, DT_INT8}},
    {VALUE_SHARED_PREFIX_NAME,    {DT_FLOAT16, DT_BF16, DT_INT8}},
    {QUERY_ROPE_NAME,             {DT_FLOAT16, DT_BF16, DT_INT8}},
    {KEY_ROPE_NAME,               {DT_FLOAT16, DT_BF16, DT_INT8}},
    {DEQUANT_SCALE_QUERY_NAME,    {DT_FLOAT}},
    {ATTEN_OUT_NAME,              {DT_FLOAT16, DT_BF16, DT_INT8}},
    {SOFTMAX_LSE_NAME,            {DT_FLOAT}},
};

const std::set<Format> FORMAT_SUPPORT_SET = {FORMAT_ND, FORMAT_NCHW, FORMAT_NHWC, FORMAT_NCDHW};
}
} // namespace optiling

#endif // FUSED_INFER_ATTENTION_SCORE_TILING_CONSTANTS_H